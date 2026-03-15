#!/usr/bin/env python3
"""
Unit tests for the check-print-debug-artifacts pre-commit hook.

The hook uses ``language: pygrep`` with the pattern ``print.*(NOTE|TODO|FIXME)``
scoped to ``examples/``.  These tests verify the regex directly so the behaviour
is documented and regression-tested without requiring the full pre-commit
runtime.
"""

import re
from typing import List, Tuple

import pytest

# The pattern used in .pre-commit-config.yaml
PATTERN = re.compile(r"print.*(NOTE|TODO|FIXME)")


def matches(line: str) -> bool:
    """Return True if *line* would be flagged by the hook."""
    return bool(PATTERN.search(line))


# ---------------------------------------------------------------------------
# Positive cases — must match (hook should FAIL on these)
# ---------------------------------------------------------------------------

POSITIVE_CASES: List[Tuple[str, str]] = [
    ('print("NOTE: fix this before release")', "bare string with NOTE"),
    ('print("TODO: remove after testing")', "bare string with TODO"),
    ('print("FIXME: broken edge case")', "bare string with FIXME"),
    ('print(f"NOTE: value is {x}")', "f-string with NOTE"),
    ('print("TODO")', "bare TODO keyword only"),
    ('print("FIXME: 42")', "FIXME with trailing content"),
    ('# print("NOTE: commented out")', "commented-out print still flagged"),
]


@pytest.mark.parametrize("line,description", POSITIVE_CASES)
def test_positive_match(line: str, description: str) -> None:
    """Hook pattern must match lines that contain print + NOTE/TODO/FIXME."""
    assert matches(line), f"Expected match for {description!r}: {line!r}"


# ---------------------------------------------------------------------------
# Negative cases — must NOT match (hook should PASS on these)
# ---------------------------------------------------------------------------

NEGATIVE_CASES: List[Tuple[str, str]] = [
    ('print("hello world")', "plain print with no keyword"),
    ('log("TODO: ignored by hook")', "non-print call with TODO"),
    ('print("notice the difference")', "word containing 'note' substring"),
    ('print("noted")', "word starting with 'note'"),
    ('print("notebook")', "word containing 'note' as a prefix"),
]


@pytest.mark.parametrize("line,description", NEGATIVE_CASES)
def test_negative_no_match(line: str, description: str) -> None:
    """Hook pattern must NOT match lines that should be allowed."""
    assert not matches(line), f"Unexpected match for {description!r}: {line!r}"
