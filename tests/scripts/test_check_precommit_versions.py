#!/usr/bin/env python3
"""Tests for check_precommit_versions.py.

Tests cover:
- parse_precommit_revs: extracting rev: values from YAML text
- parse_lock_versions: extracting versions from pixi.lock text
- normalize_rev: stripping leading 'v' prefix
- check_drift: comparing revs against lock versions
- main: end-to-end with tmp_path fixtures
"""

import sys
from pathlib import Path


# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))
from check_precommit_versions import (
    REPO_TO_PACKAGE,
    check_drift,
    main,
    normalize_rev,
    parse_lock_versions,
    parse_precommit_revs,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

PRECOMMIT_MYPY_MATCH = """\
repos:
  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.19.1
    hooks:
      - id: mypy
"""

PRECOMMIT_NBSTRIPOUT_MATCH = """\
repos:
  - repo: https://github.com/kynan/nbstripout
    rev: 0.7.1
    hooks:
      - id: nbstripout
"""

PRECOMMIT_BOTH_MATCH = """\
repos:
  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.19.1
    hooks:
      - id: mypy
  - repo: https://github.com/kynan/nbstripout
    rev: 0.7.1
    hooks:
      - id: nbstripout
"""

PRECOMMIT_MYPY_MISMATCH = """\
repos:
  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.18.0
    hooks:
      - id: mypy
"""

PRECOMMIT_UNTRACKED_ONLY = """\
repos:
  - repo: https://github.com/DavidAnson/markdownlint-cli2
    rev: v0.12.1
    hooks:
      - id: markdownlint-cli2
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.5.0
    hooks:
      - id: trailing-whitespace
"""

PRECOMMIT_LOCAL_ONLY = """\
repos:
  - repo: local
    hooks:
      - id: mojo-format
"""

LOCK_WITH_MYPY = """\
- conda: https://conda.anaconda.org/conda-forge/linux-64/mypy-1.19.1-py314h5bd0f2a_0.conda
  sha256: abc123
"""

LOCK_WITH_MYPY_DIFFERENT = """\
- conda: https://conda.anaconda.org/conda-forge/linux-64/mypy-1.20.0-py314h5bd0f2a_0.conda
  sha256: abc123
"""

LOCK_WITH_NBSTRIPOUT = """\
- conda: https://conda.anaconda.org/conda-forge/noarch/nbstripout-0.7.1-pyhd8ed1ab_0.conda
  sha256: def456
"""

LOCK_WITH_BOTH = """\
- conda: https://conda.anaconda.org/conda-forge/linux-64/mypy-1.19.1-py314h5bd0f2a_0.conda
  sha256: abc123
- conda: https://conda.anaconda.org/conda-forge/noarch/nbstripout-0.7.1-pyhd8ed1ab_0.conda
  sha256: def456
"""

LOCK_EMPTY = ""


# ── TestParsePrecommitRevs ─────────────────────────────────────────────────────


class TestParsePrecommitRevs:
    def test_extracts_mypy_rev(self, tmp_path: Path) -> None:
        config = tmp_path / ".pre-commit-config.yaml"
        config.write_text(PRECOMMIT_MYPY_MATCH)
        revs = parse_precommit_revs(config)
        assert "https://github.com/pre-commit/mirrors-mypy" in revs
        assert revs["https://github.com/pre-commit/mirrors-mypy"] == "v1.19.1"

    def test_extracts_nbstripout_rev(self, tmp_path: Path) -> None:
        config = tmp_path / ".pre-commit-config.yaml"
        config.write_text(PRECOMMIT_NBSTRIPOUT_MATCH)
        revs = parse_precommit_revs(config)
        assert "https://github.com/kynan/nbstripout" in revs
        assert revs["https://github.com/kynan/nbstripout"] == "0.7.1"

    def test_extracts_both_repos(self, tmp_path: Path) -> None:
        config = tmp_path / ".pre-commit-config.yaml"
        config.write_text(PRECOMMIT_BOTH_MATCH)
        revs = parse_precommit_revs(config)
        assert len(revs) == 2
        assert "https://github.com/pre-commit/mirrors-mypy" in revs
        assert "https://github.com/kynan/nbstripout" in revs

    def test_skips_local_repos(self, tmp_path: Path) -> None:
        config = tmp_path / ".pre-commit-config.yaml"
        config.write_text(PRECOMMIT_LOCAL_ONLY)
        revs = parse_precommit_revs(config)
        assert len(revs) == 0

    def test_empty_config(self, tmp_path: Path) -> None:
        config = tmp_path / ".pre-commit-config.yaml"
        config.write_text("repos: []\n")
        revs = parse_precommit_revs(config)
        assert revs == {}

    def test_untracked_repos_included(self, tmp_path: Path) -> None:
        """Untracked repos are still extracted; check_drift filters them."""
        config = tmp_path / ".pre-commit-config.yaml"
        config.write_text(PRECOMMIT_UNTRACKED_ONLY)
        revs = parse_precommit_revs(config)
        assert len(revs) == 2


# ── TestParseLockVersions ─────────────────────────────────────────────────────


class TestParseLockVersions:
    def test_extracts_mypy_version(self, tmp_path: Path) -> None:
        lock = tmp_path / "pixi.lock"
        lock.write_text(LOCK_WITH_MYPY)
        versions = parse_lock_versions(lock)
        assert versions.get("mypy") == "1.19.1"

    def test_extracts_nbstripout_version(self, tmp_path: Path) -> None:
        lock = tmp_path / "pixi.lock"
        lock.write_text(LOCK_WITH_NBSTRIPOUT)
        versions = parse_lock_versions(lock)
        assert versions.get("nbstripout") == "0.7.1"

    def test_extracts_both(self, tmp_path: Path) -> None:
        lock = tmp_path / "pixi.lock"
        lock.write_text(LOCK_WITH_BOTH)
        versions = parse_lock_versions(lock)
        assert versions.get("mypy") == "1.19.1"
        assert versions.get("nbstripout") == "0.7.1"

    def test_empty_lock(self, tmp_path: Path) -> None:
        lock = tmp_path / "pixi.lock"
        lock.write_text(LOCK_EMPTY)
        versions = parse_lock_versions(lock)
        assert versions == {}

    def test_no_duplicate_entries(self, tmp_path: Path) -> None:
        """First occurrence wins for duplicate package names."""
        lock_text = LOCK_WITH_MYPY + LOCK_WITH_MYPY_DIFFERENT
        lock = tmp_path / "pixi.lock"
        lock.write_text(lock_text)
        versions = parse_lock_versions(lock)
        assert versions.get("mypy") == "1.19.1"


# ── TestNormalizeRev ──────────────────────────────────────────────────────────


class TestNormalizeRev:
    def test_strips_leading_v(self) -> None:
        assert normalize_rev("v1.19.1") == "1.19.1"

    def test_no_v_unchanged(self) -> None:
        assert normalize_rev("1.19.1") == "1.19.1"

    def test_empty_string(self) -> None:
        assert normalize_rev("") == ""

    def test_only_v(self) -> None:
        assert normalize_rev("v") == ""

    def test_multiple_v_stripped(self) -> None:
        assert normalize_rev("vv1.0") == "1.0"


# ── TestCheckDrift ────────────────────────────────────────────────────────────


class TestCheckDrift:
    def test_no_mismatches_when_versions_match(self) -> None:
        revs = {"https://github.com/pre-commit/mirrors-mypy": "v1.19.1"}
        lock_versions = {"mypy": "1.19.1"}
        mismatches = check_drift(revs, lock_versions)
        assert mismatches == []

    def test_detects_mypy_mismatch(self) -> None:
        revs = {"https://github.com/pre-commit/mirrors-mypy": "v1.18.0"}
        lock_versions = {"mypy": "1.19.1"}
        mismatches = check_drift(revs, lock_versions)
        assert len(mismatches) == 1
        repo_url, precommit_ver, lock_ver = mismatches[0]
        assert "mirrors-mypy" in repo_url
        assert precommit_ver == "1.18.0"
        assert lock_ver == "1.19.1"

    def test_detects_nbstripout_mismatch(self) -> None:
        revs = {"https://github.com/kynan/nbstripout": "0.6.0"}
        lock_versions = {"nbstripout": "0.7.1"}
        mismatches = check_drift(revs, lock_versions)
        assert len(mismatches) == 1
        assert mismatches[0][1] == "0.6.0"
        assert mismatches[0][2] == "0.7.1"

    def test_skips_untracked_repos(self) -> None:
        revs = {"https://github.com/DavidAnson/markdownlint-cli2": "v0.12.1"}
        lock_versions = {"mypy": "1.19.1"}
        mismatches = check_drift(revs, lock_versions)
        assert mismatches == []

    def test_skips_when_package_not_in_lock(self) -> None:
        revs = {"https://github.com/pre-commit/mirrors-mypy": "v1.19.1"}
        lock_versions = {}  # mypy not in lock
        mismatches = check_drift(revs, lock_versions)
        assert mismatches == []

    def test_multiple_mismatches(self) -> None:
        revs = {
            "https://github.com/pre-commit/mirrors-mypy": "v1.18.0",
            "https://github.com/kynan/nbstripout": "0.6.0",
        }
        lock_versions = {"mypy": "1.19.1", "nbstripout": "0.7.1"}
        mismatches = check_drift(revs, lock_versions)
        assert len(mismatches) == 2

    def test_empty_revs(self) -> None:
        mismatches = check_drift({}, {"mypy": "1.19.1"})
        assert mismatches == []

    def test_empty_lock_versions(self) -> None:
        revs = {"https://github.com/pre-commit/mirrors-mypy": "v1.19.1"}
        mismatches = check_drift(revs, {})
        assert mismatches == []


# ── TestMainExitCodes ─────────────────────────────────────────────────────────


class TestMainExitCodes:
    def _write_config(self, tmp_path: Path, content: str) -> None:
        (tmp_path / ".pre-commit-config.yaml").write_text(content)

    def _write_lock(self, tmp_path: Path, content: str) -> None:
        (tmp_path / "pixi.lock").write_text(content)

    def test_returns_0_when_versions_match(self, tmp_path: Path) -> None:
        self._write_config(tmp_path, PRECOMMIT_MYPY_MATCH)
        self._write_lock(tmp_path, LOCK_WITH_MYPY)
        assert main(["--repo-root", str(tmp_path)]) == 0

    def test_returns_1_when_versions_mismatch(self, tmp_path: Path) -> None:
        self._write_config(tmp_path, PRECOMMIT_MYPY_MISMATCH)
        self._write_lock(tmp_path, LOCK_WITH_MYPY)
        assert main(["--repo-root", str(tmp_path)]) == 1

    def test_returns_0_when_no_tracked_repos(self, tmp_path: Path) -> None:
        self._write_config(tmp_path, PRECOMMIT_UNTRACKED_ONLY)
        self._write_lock(tmp_path, LOCK_WITH_MYPY)
        assert main(["--repo-root", str(tmp_path)]) == 0

    def test_returns_0_when_nbstripout_matches(self, tmp_path: Path) -> None:
        self._write_config(tmp_path, PRECOMMIT_NBSTRIPOUT_MATCH)
        self._write_lock(tmp_path, LOCK_WITH_NBSTRIPOUT)
        assert main(["--repo-root", str(tmp_path)]) == 0

    def test_returns_0_when_both_match(self, tmp_path: Path) -> None:
        self._write_config(tmp_path, PRECOMMIT_BOTH_MATCH)
        self._write_lock(tmp_path, LOCK_WITH_BOTH)
        assert main(["--repo-root", str(tmp_path)]) == 0


# ── TestMissingFiles ──────────────────────────────────────────────────────────


class TestMissingFiles:
    def test_returns_1_when_config_missing(self, tmp_path: Path) -> None:
        (tmp_path / "pixi.lock").write_text(LOCK_WITH_MYPY)
        assert main(["--repo-root", str(tmp_path)]) == 1

    def test_returns_1_when_lock_missing(self, tmp_path: Path) -> None:
        (tmp_path / ".pre-commit-config.yaml").write_text(PRECOMMIT_MYPY_MATCH)
        assert main(["--repo-root", str(tmp_path)]) == 1

    def test_returns_1_when_both_missing(self, tmp_path: Path) -> None:
        assert main(["--repo-root", str(tmp_path)]) == 1


# ── TestRepoToPackageMapping ──────────────────────────────────────────────────


class TestRepoToPackageMapping:
    def test_mirrors_mypy_mapped(self) -> None:
        assert "mirrors-mypy" in REPO_TO_PACKAGE
        assert REPO_TO_PACKAGE["mirrors-mypy"] == "mypy"

    def test_nbstripout_mapped(self) -> None:
        assert "kynan/nbstripout" in REPO_TO_PACKAGE
        assert REPO_TO_PACKAGE["kynan/nbstripout"] == "nbstripout"
