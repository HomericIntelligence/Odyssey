#!/usr/bin/env python3
"""Regression test: all 'uses:' lines in .github/ must be SHA-pinned.

Asserts:
- Every 'uses: owner/repo@<ref>' line contains a 40-char hex SHA (not a tag).
- Every SHA-pinned line also contains a '#' comment with human-readable version.
- Every workflow YAML file is valid YAML.
"""

import re
from pathlib import Path
from typing import List

import pytest
import yaml

GITHUB_DIR = Path(__file__).parents[2] / ".github"
WORKFLOW_FILES: List[Path] = sorted(
    list(GITHUB_DIR.glob("workflows/*.yml")) + list(GITHUB_DIR.glob("actions/**/*.yml"))
)

SHA_RE = re.compile(r"uses:\s+\S+@([0-9a-f]{40})")
TAG_RE = re.compile(r"uses:\s+\S+@v[0-9]")
COMMENT_RE = re.compile(r"uses:\s+\S+@[0-9a-f]{40}.*#")


@pytest.mark.parametrize("workflow_file", WORKFLOW_FILES, ids=lambda f: f.name)
def test_workflow_yaml_is_valid(workflow_file: Path) -> None:
    """Every workflow YAML file must be parseable by yaml.safe_load."""
    yaml.safe_load(workflow_file.read_text())


@pytest.mark.parametrize("workflow_file", WORKFLOW_FILES, ids=lambda f: f.name)
def test_no_tag_pinned_actions(workflow_file: Path) -> None:
    """No 'uses:' line may reference an action by version tag (e.g. @v3)."""
    for i, line in enumerate(workflow_file.read_text().splitlines(), 1):
        assert not TAG_RE.search(line), (
            f"{workflow_file.name}:{i}: tag-pinned action found (use SHA instead): {line.strip()}"
        )


@pytest.mark.parametrize("workflow_file", WORKFLOW_FILES, ids=lambda f: f.name)
def test_sha_pinned_actions_have_version_comment(workflow_file: Path) -> None:
    """Every SHA-pinned 'uses:' line must include a '#' comment for human readability."""
    for i, line in enumerate(workflow_file.read_text().splitlines(), 1):
        # Skip local composite action references (e.g. uses: ./.github/actions/setup-pixi)
        if SHA_RE.search(line) and "./.github" not in line:
            assert COMMENT_RE.search(line), (
                f"{workflow_file.name}:{i}: SHA-pinned action missing version comment: {line.strip()}"
            )
