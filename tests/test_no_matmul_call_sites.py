#!/usr/bin/env python3
"""
Unit tests for the no-matmul-call-sites pre-commit hook.

The hook's bash grep chain is:

.. code-block:: bash

    grep -rn '.__matmul__(' . ... |
      grep -v 'fn __matmul__(' |
      grep -v '# __matmul__' |
      grep -v '__matmul__.*deprecated'


A line is a **violation** when it:
  1. Contains ``.__matmul__(``
  2. Does NOT contain ``fn __matmul__(``
  3. Does NOT contain ``# __matmul__``
  4. Does NOT contain ``__matmul__.*deprecated``

Tests mirror the hook logic via pure Python ``re`` so they run without
subprocess or pre-commit invocation, ensuring fast regression testing.
"""

import re

import pytest


# ---------------------------------------------------------------------------
# Hook logic predicate
# ---------------------------------------------------------------------------


def is_violation(line: str) -> bool:
    """Return True if *line* would be flagged by the no-matmul-call-sites hook.

    Replicates the bash grep chain::

        grep '.__matmul__('
        grep -v 'fn __matmul__('
        grep -v '# __matmul__'
        grep -v '__matmul__.*deprecated'
    """
    if not re.search(r"\.__matmul__\(", line):
        return False
    if re.search(r"fn __matmul__\(", line):
        return False
    if re.search(r"#\s*__matmul__", line):
        return False
    if re.search(r"__matmul__.*deprecated", line):
        return False
    return True


# ---------------------------------------------------------------------------
# Positive cases — lines that SHOULD be caught
# ---------------------------------------------------------------------------


class TestPositiveCases:
    @pytest.mark.parametrize(
        "line",
        [
            "    result = a.__matmul__(b)",
            "    c = self.__matmul__(other)",
            "output = x.__matmul__(y)",
            "    return lhs.__matmul__(rhs)",
        ],
    )
    def test_call_site_is_violation(self, line: str) -> None:
        """Direct .__matmul__() call sites must be flagged."""
        assert is_violation(line), f"Expected violation for: {line!r}"


# ---------------------------------------------------------------------------
# Negative cases — lines that SHOULD be excluded
# ---------------------------------------------------------------------------


class TestNegativeCases:
    def test_function_definition_excluded(self) -> None:
        """Method definition ``fn __matmul__(self, …)`` is not a call site."""
        line = "fn __matmul__(self, rhs: Self) -> Self:"
        assert not is_violation(line)

    def test_comment_line_excluded(self) -> None:
        """A full comment containing ``# __matmul__`` is excluded."""
        line = "# __matmul__ is deprecated"
        assert not is_violation(line)

    def test_indented_comment_excluded(self) -> None:
        """An indented comment is also excluded."""
        line = "    # __matmul__ call is forbidden"
        assert not is_violation(line)

    @pytest.mark.parametrize(
        "line",
        [
            "# __matmul__ is deprecated, use matmul() instead",
            "    # __matmul__.*deprecated pattern",
        ],
    )
    def test_deprecated_comment_variants_excluded(self, line: str) -> None:
        """Comment lines about deprecation are excluded."""
        assert not is_violation(line)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_string_literal_is_caught(self) -> None:
        """The hook has no AST awareness — ``.__matmul__(`` inside a string
        literal IS flagged as a false positive.  This test documents the
        known limitation rather than asserting ideal behaviour."""
        line = 'var msg = ".__matmul__( is a call site"'
        # grep cannot distinguish string contents; the hook fires here
        assert is_violation(line)

    def test_function_def_with_body_on_same_line_excluded(self) -> None:
        """A rare single-line fn that also has .__matmul__( later is
        excluded because ``fn __matmul__(`` appears first."""
        line = "fn __matmul__(self, rhs: Self) -> Self: return self.__matmul__(rhs)"
        # grep -v "fn __matmul__(" removes this whole line
        assert not is_violation(line)

    def test_no_matmul_in_line_not_violation(self) -> None:
        """Lines without .__matmul__( at all are never violations."""
        line = "    result = matmul(a, b)"
        assert not is_violation(line)

    def test_matmul_without_dot_not_violation(self) -> None:
        """``__matmul__(`` without a leading dot is not a call site."""
        line = "    x = __matmul__(a, b)"
        assert not is_violation(line)
