#!/usr/bin/env python3
"""Regression tests for security.yml workflow hardening.

Asserts that:
- Gitleaks binary download includes SHA256 checksum verification
- The SHA256 value is a real 64-character hex digest (not a placeholder)
- Gitleaks is pinned to the expected version
- fetch-depth: 0 is set in the secret-scan job checkout step
"""

import re
from pathlib import Path

import pytest

WORKFLOW = Path(__file__).parents[2] / ".github" / "workflows" / "security.yml"
GITLEAKS_VERSION = "v8.30.0"
GITLEAKS_TARBALL = "gitleaks_8.30.0_linux_x64.tar.gz"
EXPECTED_SHA256 = "79a3ab579b53f71efd634f3aaf7e04a0fa0cf206b7ed434638d1547a2470a66e"


@pytest.fixture(scope="module")
def workflow_content() -> str:
    """Read the workflow file once for all tests."""
    return WORKFLOW.read_text()


def test_workflow_file_exists() -> None:
    """Verify the workflow file exists at the expected path."""
    assert WORKFLOW.exists(), f"Workflow file not found at {WORKFLOW}"


def test_gitleaks_version_pinned(workflow_content: str) -> None:
    """Gitleaks must be pinned to a specific version in the download URL."""
    assert GITLEAKS_VERSION in workflow_content, f"Gitleaks must be pinned to {GITLEAKS_VERSION} in the download URL"


def test_gitleaks_sha256_check_present(workflow_content: str) -> None:
    """SHA256 checksum verification must be present before tar extraction."""
    assert "sha256sum --check" in workflow_content, (
        "sha256sum --check must be present to verify Gitleaks binary integrity before execution"
    )


def test_sha256_value_is_real_hex_digest(workflow_content: str) -> None:
    """The SHA256 value must be a real 64-char hex digest, not a placeholder."""
    match = re.search(r"([0-9a-f]{64})\s+" + re.escape(GITLEAKS_TARBALL), workflow_content)
    assert match is not None, (
        f"Must contain a real 64-character hex SHA256 digest for {GITLEAKS_TARBALL}, not a placeholder"
    )


def test_sha256_value_matches_expected(workflow_content: str) -> None:
    """The hardcoded SHA256 must match the official checksum for gitleaks_8.30.0_linux_x64.tar.gz."""
    match = re.search(r"([0-9a-f]{64})\s+" + re.escape(GITLEAKS_TARBALL), workflow_content)
    assert match is not None, f"SHA256 line for {GITLEAKS_TARBALL} not found"
    actual = match.group(1)
    assert actual == EXPECTED_SHA256, (
        f"SHA256 mismatch for {GITLEAKS_TARBALL}:\n"
        f"  expected: {EXPECTED_SHA256}\n"
        f"  actual:   {actual}\n"
        "Update EXPECTED_SHA256 in this test when upgrading gitleaks."
    )


def test_fetch_depth_zero_present(workflow_content: str) -> None:
    """fetch-depth: 0 must be set in the secret-scan job checkout step."""
    assert "fetch-depth: 0" in workflow_content, (
        "fetch-depth: 0 must be set in the secret-scan job to ensure full git history is available"
    )
