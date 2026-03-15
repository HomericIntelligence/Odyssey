#!/usr/bin/env python3
"""Unit tests for the no-matmul-call-sites pre-commit hook.

Verifies that the grep chain correctly:
1. Catches positive: a.__matmul__(b) call sites
2. Excludes negative: fn __matmul__(...) method definitions
3. Excludes negative: # __matmul__ comments
4. Handles edge case: .__matmul__( inside string literals

Follow-up from #3215.
"""

from __future__ import annotations


# Recreate the grep chain logic from the hook
# Entry: grep -rn "\.__matmul__(" . | grep -v "fn __matmul__(" | grep -v "# __matmul__"


def _matches_hook_pattern(line: str) -> bool:
    """Check if a line would be caught by the no-matmul hook.

    Returns True if the line contains a call site .__matmul__(
    AND does not match any of the exclusion patterns.
    """
    # Must contain .__matmul__(
    if r".__matmul__(" not in line:
        return False

    # Exclude method definitions
    if "fn __matmul__(" in line:
        return False

    # Exclude comments
    if "# __matmul__" in line:
        return False

    return True


def test_catch_matmul_call_site():
    """Positive: a.__matmul__(b) is caught by the hook."""
    line = "    result = a.__matmul__(b)  # VIOLATION"
    assert _matches_hook_pattern(line), "Should catch .__matmul__() call site"


def test_catch_matmul_call_site_nested():
    """Positive: .__matmul__() in nested context is caught."""
    line = "    output = (matrix1.__matmul__(matrix2) + bias)"
    assert _matches_hook_pattern(line), "Should catch nested .__matmul__() call"


def test_exclude_method_definition():
    """Negative: fn __matmul__() method definition is excluded."""
    line = "    fn __matmul__(self, rhs: Self) -> Self:"
    assert not _matches_hook_pattern(line), "Should exclude fn __matmul__() definition"


def test_exclude_method_definition_multiline():
    """Negative: fn __matmul__() with full signature is excluded."""
    line = "fn __matmul__(mut self, other: Self) raises -> Self:"
    assert not _matches_hook_pattern(line), "Should exclude method definition"


def test_exclude_comment():
    """Negative: # __matmul__ comment is excluded."""
    line = "    # __matmul__ is deprecated, use matmul() instead"
    assert not _matches_hook_pattern(line), "Should exclude comments about __matmul__"


def test_exclude_todo_comment():
    """Negative: TODO with __matmul__ comment is excluded."""
    line = "    # TODO: Remove __matmul__ calls from this file"
    assert not _matches_hook_pattern(line), "Should exclude TODO comments"


def test_edge_case_string_literal():
    """Edge case: .__matmul__( inside string might match but is acceptable.

    The hook pattern is permissive (grep-based) and will match string literals
    containing .__matmul__(. This is acceptable because:
    1. String literals with .__matmul__() are rare
    2. False positives can be fixed with # __matmul__ comment
    3. Keeping the pattern simple is better than complex regex
    """
    line = '    docstring = """use .__matmul__( instead of @ operator"""'
    # This will match, which is technically a false positive, but acceptable
    assert _matches_hook_pattern(line), "String literals will match (acceptable)"


def test_edge_case_string_in_source():
    """Edge case: Escaped string with .__matmul__( matches but is rare."""
    line = '    error_msg = "Call .__matmul__( from tensor module"'
    assert _matches_hook_pattern(line), "String with .__matmul__( will match"


def test_no_false_negatives():
    """Verify common real-world matmul call patterns are caught."""
    patterns = [
        "result = tensor1.__matmul__(tensor2)",
        "output = self.weight.__matmul__(input)",
        "x = matrix_a.__matmul__(matrix_b)  # multiplication",
        "data.__matmul__(other)",
        "    a.__matmul__(b)",
    ]
    for pattern in patterns:
        assert _matches_hook_pattern(pattern), f"Should catch: {pattern}"


def test_no_false_positives_on_excluded_patterns():
    """Verify no matches on valid exclusion patterns."""
    patterns = [
        "fn __matmul__(self, other: Self) -> Self:",
        "def __matmul__(self, other):",
        "# __matmul__ operator is disabled",
        "# TODO: optimize __matmul__ call",
        "    # Fix: __matmul__ should use SIMD",
    ]
    for pattern in patterns:
        assert not _matches_hook_pattern(pattern), f"Should not match: {pattern}"
