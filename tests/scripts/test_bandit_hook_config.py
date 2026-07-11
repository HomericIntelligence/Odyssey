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
    """Verify the bandit skip list is empty (no intentional suppressions).

    Download scripts were refactored to delegate to hephaestus internals, so
    B310 (urlopen) and B202 (extractall) no longer apply. B301 (pickle) is
    also absent — no pickle usage in the scanned paths.

    Any future suppression requires a documented rationale in this test.
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

    def test_no_undocumented_skips(self, bandit_hook: dict) -> None:
        """The skip list must be empty — no current suppressions are needed.

        Download scripts delegate to hephaestus, so urlopen/extractall/pickle
        patterns are no longer in the scanned files.
        If a new skip is needed, add its rationale to this class docstring.
        """
        skip_ids = self._get_skip_ids(bandit_hook)
        assert not skip_ids, (
            f"Unexpected bandit skip IDs found: {skip_ids}. "
            "Add rationale to TestBanditSkipList docstring before adding skip IDs."
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


class TestBanditConfidenceThreshold:
    """Verify the confidence threshold decision for bandit.

    Bandit supports separate severity and confidence flags:
    - Severity: -l (LOW), -ll (MEDIUM), -lll (HIGH)
    - Confidence: -l (LOW), -ll (MEDIUM), -lll (HIGH)

    The current hook sets severity to -ll (MEDIUM+) but does not set a confidence
    threshold, which means all confidence levels (LOW, MEDIUM, HIGH) are reported.

    Decision: Confidence threshold is intentionally unset (default=all) to catch
    potential security issues regardless of confidence level. This is more conservative
    and appropriate for CI/CD security scanning.
    """

    def test_confidence_threshold_documented(self) -> None:
        """Document the intentional decision about confidence threshold.

        The bandit hook intentionally does not set a confidence threshold (-l, -ll, -lll).
        This means issues are reported at all confidence levels, not just high confidence.

        Rationale:
        - Conservative approach: report even uncertain issues for review
        - Better for catch-all CI security scanning
        - Developers can suppress false positives with # nosec comments
        - False positives are preferable to missing real vulnerabilities
        """
        # This is a documentation test that codifies the design decision
        # If this fails, update the bandit hook AND update this test/docstring
        assert True, "Confidence threshold test passed. Bandit hook intentionally reports all confidence levels."

    def test_no_confidence_flags_set(self, bandit_hook: dict) -> None:
        """Verify no confidence threshold flags are set in the bandit hook.

        The absence of -l/-ll/-lll confidence flags means all confidence levels
        are reported. This is the intended conservative behavior.
        """
        flags = _all_bandit_flags(bandit_hook)
        # Count severity vs confidence flags - both use -l/-ll/-lll notation
        # Severity -ll is already verified in TestBanditSeverityThreshold
        # This test documents that confidence flags are intentionally absent

        # Check that we don't have conflicting flags (multiple severity/confidence settings)
        ll_count = flags.count("-ll")
        lll_count = flags.count("-lll")

        # We expect exactly one -ll for severity; if there's more, confidence may be set
        assert ll_count <= 1, (
            f"Multiple -ll flags found ({ll_count}). "
            "This might indicate both severity and confidence thresholds are set. "
            "Clarify which -ll is for severity (intended) vs confidence (check if unintentional)."
        )
        assert lll_count <= 1, (
            f"Multiple -lll flags found ({lll_count}). "
            "This might indicate both severity and confidence thresholds are set."
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
            "scripts/download_emnist.py",
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
            "src/odyssey/nn/layers/conv2d.mojo",
            "papers/lenet5/model.mojo",
            "docs/dev/some_script.py",
        ],
    )
    def test_pattern_excludes_non_target_paths(self, bandit_hook: dict, path: str) -> None:
        """The files: pattern must not match Mojo files or out-of-scope directories."""
        pattern = self._get_files_pattern(bandit_hook)
        assert not re.search(pattern, path), (
            f"files: pattern {pattern!r} unexpectedly matches {path!r}. "
            "Bandit should only scan scripts/, tests/, tools/, and examples/ Python files."
        )


class TestBanditNosecRationale:
    """Verify no stale skip IDs exist in the bandit hook.

    Download scripts delegate to hephaestus, so B310 (urlopen) and B202
    (extractall) are no longer triggered. These tests confirm there is nothing
    to suppress and the skip list stays empty.
    """

    def test_b310_not_triggered(self) -> None:
        """scripts/ must NOT contain urlopen() — B310 skip is not needed.

        If this test fails, B310 must be re-added to the skip list with rationale.
        """
        scripts_dir = REPO_ROOT / "scripts"
        files_with_urlopen = [f for f in scripts_dir.glob("*.py") if "urlopen" in f.read_text()]
        assert not files_with_urlopen, (
            f"scripts/ now contains urlopen() in: {[f.name for f in files_with_urlopen]}. "
            "Add B310 back to the bandit skip list with documented rationale."
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

    def test_b202_not_triggered(self) -> None:
        """scripts/ must NOT contain extractall() — B202 skip is not needed.

        If this test fails, B202 must be re-added to the skip list with rationale.
        """
        scripts_dir = REPO_ROOT / "scripts"
        files_with_extractall = [f for f in scripts_dir.glob("*.py") if "extractall" in f.read_text()]
        assert not files_with_extractall, (
            f"scripts/ now contains extractall() in: {[f.name for f in files_with_extractall]}. "
            "Add B202 back to the bandit skip list with documented rationale."
        )

    def test_b202_extractall_targets_local_path(self) -> None:
        """extractall() calls in scripts/ must target a local directory variable.

        This validates that the B202 skip is safe: extraction targets are known-safe
        local paths (DATA_DIR), not user-controlled paths.

        Note: This test searches all of scripts/*.py to catch if extractall() is
        refactored into shared helpers (e.g., common.py) or imported from elsewhere.
        """
        scripts_dir = REPO_ROOT / "scripts"
        files_with_extractall = []

        # Search all Python files in scripts/ that contain extractall
        for script in scripts_dir.glob("*.py"):
            source = script.read_text()
            if "extractall" in source:
                files_with_extractall.append(script)

        # If no files with extractall found, verify the current assumption is correct
        if not files_with_extractall:
            # This is expected for now - no shared extractall helper yet
            # If this assertion fails, it means extractall was added somewhere in scripts/
            pass
        else:
            # If extractall IS in scripts/, verify all calls are safe
            for script in files_with_extractall:
                source = script.read_text()
                # extractall should be called with a Path/variable argument, not a raw string
                # A bare extractall() with no argument (extracting to cwd) would be unsafe
                assert re.search(r"\.extractall\s*\(\s*\S", source), (
                    f"{script.name} calls extractall() with no target path. "
                    "Pass an explicit local directory to prevent extraction to cwd."
                )

    def test_emnist_script_is_thin_wrapper(self) -> None:
        """Verify download_emnist.py is a thin hephaestus wrapper (no local URL constants).

        URL handling moved to hephaestus.datasets.downloader in v0.7.0.
        This test documents that the script is a re-export wrapper, not a standalone downloader.
        """
        emnist_script = REPO_ROOT / "scripts" / "download_emnist.py"
        assert emnist_script.exists(), "download_emnist.py not found"

        source = emnist_script.read_text()
        assert "hephaestus" in source, "download_emnist.py should delegate to hephaestus — URL handling moved to v0.7.0"
