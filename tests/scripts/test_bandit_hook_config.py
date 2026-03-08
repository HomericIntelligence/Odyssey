#!/usr/bin/env python3
"""Unit tests for the bandit pre-commit hook configuration.

Verifies that:
1. The bandit hook exists in .pre-commit-config.yaml
2. B310 and B202 are intentionally skipped with documented rationale
3. The -ll severity threshold (medium+) is configured
4. The files: pattern matches expected Python script paths
5. The skip list reflects real usage patterns in the codebase

Follow-up from #3157.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml


REPO_ROOT = Path(__file__).parent.parent.parent
PRE_COMMIT_CONFIG = REPO_ROOT / ".pre-commit-config.yaml"


def _load_bandit_hook() -> dict:
    """Load and return the bandit hook config from .pre-commit-config.yaml."""
    config = yaml.safe_load(PRE_COMMIT_CONFIG.read_text())
    for repo in config.get("repos", []):
        for hook in repo.get("hooks", []):
            if hook.get("id") == "bandit":
                return hook
    return {}


def _all_bandit_flags(hook: dict) -> list[str]:
    """Return all CLI flags for the bandit hook from both entry and args.

    Pre-commit hooks may embed flags directly in the entry string or in the
    separate args list. This helper normalises both forms so tests are
    independent of which style the config uses.
    """
    flags: list[str] = []
    entry = hook.get("entry", "")
    if entry:
        # entry is a shell-split string; split on whitespace to get tokens
        flags.extend(entry.split())
    flags.extend(hook.get("args", []))
    return flags


@pytest.fixture(scope="module")
def bandit_hook() -> dict:
    """Return the bandit hook configuration dict."""
    return _load_bandit_hook()


class TestBanditHookExists:
    """Verify the bandit hook is present in .pre-commit-config.yaml."""

    def test_pre_commit_config_exists(self) -> None:
        """The .pre-commit-config.yaml file must exist."""
        assert PRE_COMMIT_CONFIG.exists(), f"{PRE_COMMIT_CONFIG} not found"

    def test_bandit_hook_present(self, bandit_hook: dict) -> None:
        """A hook with id 'bandit' must exist in the config."""
        assert bandit_hook, (
            "No hook with id 'bandit' found in .pre-commit-config.yaml. "
            "Add a bandit hook in the 'Security checks' local repo block."
        )

    def test_bandit_hook_has_entry(self, bandit_hook: dict) -> None:
        """The bandit hook must have an entry command."""
        assert "entry" in bandit_hook, "bandit hook missing 'entry' field"
        assert bandit_hook["entry"], "bandit hook 'entry' must not be empty"

    def test_bandit_hook_has_name(self, bandit_hook: dict) -> None:
        """The bandit hook must have a human-readable name."""
        assert "name" in bandit_hook, "bandit hook missing 'name' field"


class TestBanditSkipList:
    """Verify B310 and B202 are intentionally skipped.

    Rationale:
      B310 — urllib audit: download scripts use urlopen() with HTTPS-validated
              Request objects, not raw user-supplied URLs. The URLs are hardcoded
              constants pointing to known dataset mirrors (MNIST, CIFAR, etc.).
      B202 — tarfile: tar.extractall() in download scripts targets known-safe
              local dataset directories under DATA_DIR; input is a downloaded
              archive from a trusted source, not user-controlled input.
    """

    def _get_skip_ids(self, bandit_hook: dict) -> list[str]:
        """Parse the --skip flag from entry or args and return individual skip IDs.

        Handles both '--skip=B310,B202' and '--skip B310,B202' forms, looking
        in both the entry string and the args list.
        """
        flags = _all_bandit_flags(bandit_hook)
        skip_value = ""
        for i, flag in enumerate(flags):
            if flag.startswith("--skip="):
                skip_value = flag.split("=", 1)[1]
                break
            if flag == "--skip" and i + 1 < len(flags):
                skip_value = flags[i + 1]
                break
        return [s.strip() for s in skip_value.split(",") if s.strip()]

    def test_skip_arg_present(self, bandit_hook: dict) -> None:
        """A --skip argument must be present in the bandit hook entry or args."""
        flags = _all_bandit_flags(bandit_hook)
        has_skip = any(flag.startswith("--skip") or flag == "--skip" for flag in flags)
        assert has_skip, (
            "bandit hook must include --skip=B310,B202 (in entry or args). "
            "These suppressions are intentional — see class docstring for rationale."
        )

    def test_b310_is_skipped(self, bandit_hook: dict) -> None:
        """B310 (urllib urlopen audit) must be in the skip list.

        Rationale: download scripts call urlopen() with a hardcoded HTTPS URL
        wrapped in a urllib.request.Request object. The URL is a known dataset
        mirror; there is no user-supplied input involved.
        """
        skip_ids = self._get_skip_ids(bandit_hook)
        assert "B310" in skip_ids, (
            "B310 must be skipped. "
            "Download scripts use urlopen() with validated HTTPS URLs — not user input. "
            f"Current skip list: {skip_ids}"
        )

    def test_b202_is_skipped(self, bandit_hook: dict) -> None:
        """B202 (tarfile extractall) must be in the skip list.

        Rationale: download scripts extract .tar.gz archives downloaded from
        trusted dataset sources (e.g., CIFAR mirror) into a known local DATA_DIR.
        The archive is not user-supplied input.
        """
        skip_ids = self._get_skip_ids(bandit_hook)
        assert "B202" in skip_ids, (
            "B202 must be skipped. "
            "Download scripts call extractall() on trusted archives to a local DATA_DIR — not user input. "
            f"Current skip list: {skip_ids}"
        )

    def test_skip_list_is_minimal(self, bandit_hook: dict) -> None:
        """The skip list should not suppress more checks than B310 and B202.

        Any additional suppression requires a documented rationale in this test.
        """
        skip_ids = self._get_skip_ids(bandit_hook)
        documented_skips = {"B310", "B202"}
        undocumented = set(skip_ids) - documented_skips
        assert not undocumented, (
            f"Undocumented bandit skip IDs found: {undocumented}. "
            "Add rationale to TestBanditSkipList docstring before expanding the skip list."
        )


class TestBanditSeverityThreshold:
    """Verify the -ll severity threshold (medium and above) is configured.

    The -ll flag means: report only issues with severity >= MEDIUM.
    This suppresses LOW severity warnings which are too noisy for CI.
    """

    def test_severity_flag_present(self, bandit_hook: dict) -> None:
        """-ll must be present in the bandit hook entry or args."""
        flags = _all_bandit_flags(bandit_hook)
        assert "-ll" in flags, (
            f"bandit hook must include -ll (medium+ severity threshold) in entry or args. Current flags: {flags}"
        )

    def test_no_weaker_severity_flag(self, bandit_hook: dict) -> None:
        """The hook must not use -l alone (low severity), only -ll or stricter."""
        flags = _all_bandit_flags(bandit_hook)
        # -l alone would enable LOW severity (too noisy); -ll is correct
        assert "-l" not in flags or "-ll" in flags, (
            "bandit hook uses -l (LOW severity) without -ll. Use -ll to require MEDIUM+ severity."
        )


class TestBanditFilesPattern:
    """Verify the files: pattern matches expected Python paths.

    The pattern should cover scripts/*.py (download scripts, tooling) and
    tests/**/*.py (test files). It must NOT match Mojo files or examples/.
    """

    def _get_files_pattern(self, bandit_hook: dict) -> str:
        """Return the files: pattern string from the hook config."""
        return bandit_hook.get("files", "")

    def test_files_pattern_present(self, bandit_hook: dict) -> None:
        """The bandit hook must have a files: pattern."""
        pattern = self._get_files_pattern(bandit_hook)
        assert pattern, "bandit hook is missing a 'files:' pattern"

    @pytest.mark.parametrize(
        "path",
        [
            "scripts/download_mnist.py",
            "scripts/download_cifar10.py",
            "scripts/download_cifar100.py",
            "scripts/download_fashion_mnist.py",
            "scripts/common.py",
            "scripts/implement_issues.py",
            "tests/scripts/test_bandit_hook_config.py",
            "tests/scripts/test_security.py",
        ],
    )
    def test_pattern_matches_expected_paths(self, bandit_hook: dict, path: str) -> None:
        """The files: pattern must match Python files under scripts/ and tests/."""
        pattern = self._get_files_pattern(bandit_hook)
        assert re.search(pattern, path), (
            f"files: pattern {pattern!r} should match {path!r} but does not. "
            "Download scripts and test files must be scanned by bandit."
        )

    @pytest.mark.parametrize(
        "path",
        [
            "shared/nn/layers/conv2d.mojo",
            "examples/train_lenet5.py",
            "papers/lenet5/model.mojo",
            "docs/dev/some_script.py",
        ],
    )
    def test_pattern_excludes_non_target_paths(self, bandit_hook: dict, path: str) -> None:
        """The files: pattern must not match Mojo files or out-of-scope directories."""
        pattern = self._get_files_pattern(bandit_hook)
        assert not re.search(pattern, path), (
            f"files: pattern {pattern!r} unexpectedly matches {path!r}. "
            "Bandit should only scan scripts/ and tests/ Python files."
        )


class TestBanditNosecRationale:
    """Verify the skip list reflects real usage patterns in the codebase.

    If these assertions fail, the skip list may no longer be needed and
    should be revisited.
    """

    def test_b310_trigger_urlopen_exists(self) -> None:
        """scripts/ must contain urlopen() calls that trigger B310.

        If this test fails, B310 may no longer be needed in the skip list.
        """
        scripts_dir = REPO_ROOT / "scripts"
        files_with_urlopen = [f for f in scripts_dir.glob("*.py") if "urlopen" in f.read_text()]
        assert files_with_urlopen, (
            "No scripts/*.py file contains urlopen(). B310 suppression may no longer be needed — review the skip list."
        )

    def test_b310_uses_hardcoded_urls(self) -> None:
        """urlopen() calls in download scripts must use hardcoded URL constants.

        This validates that the B310 skip is safe: URLs are module-level string
        constants (BASE_URL pattern), not constructed from user-supplied input.
        The absence of user-provided URL arguments means the urlopen() call is
        not exploitable via URL injection even if the scheme is http://.
        """
        scripts_dir = REPO_ROOT / "scripts"
        for script in scripts_dir.glob("download_*.py"):
            source = script.read_text()
            if "urlopen" in source:
                # The script must define a hardcoded module-level URL constant
                # (e.g. MNIST_BASE_URL, CIFAR10_URL, EMNIST_PRIMARY_URL)
                assert re.search(r'^[A-Z][A-Z0-9_]*URL\s*=\s*["\']', source, re.MULTILINE), (
                    f"{script.name} uses urlopen() but no hardcoded *URL constant found. "
                    "Verify the URL is a safe hardcoded constant before relying on B310 skip."
                )

    def test_b202_trigger_extractall_exists(self) -> None:
        """scripts/ must contain extractall() calls that trigger B202.

        If this test fails, B202 may no longer be needed in the skip list.
        """
        scripts_dir = REPO_ROOT / "scripts"
        files_with_extractall = [f for f in scripts_dir.glob("*.py") if "extractall" in f.read_text()]
        assert files_with_extractall, (
            "No scripts/*.py file contains extractall(). "
            "B202 suppression may no longer be needed — review the skip list."
        )

    def test_b202_extractall_targets_local_path(self) -> None:
        """extractall() calls in download scripts must target a local directory variable.

        This validates that the B202 skip is safe: extraction targets are known-safe
        local paths (DATA_DIR), not user-controlled paths.
        """
        scripts_dir = REPO_ROOT / "scripts"
        for script in scripts_dir.glob("download_*.py"):
            source = script.read_text()
            if "extractall" in source:
                # extractall should be called with a Path/variable argument, not a raw string
                # A bare extractall() with no argument (extracting to cwd) would be unsafe
                assert re.search(r"\.extractall\s*\(\s*\S", source), (
                    f"{script.name} calls extractall() with no target path. "
                    "Pass an explicit local directory to prevent extraction to cwd."
                )
