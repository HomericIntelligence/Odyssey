"""Tests for integer assertion functions.

# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_assertions.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md

Note: Split from test_assertions_float.mojo to satisfy ADR-009 ≤8
fn test_ target per file.
"""

from testing import assert_true
from shared.testing.assertions import (
    assert_equal_int,
)


fn test_assert_equal_int_specialized_passes() raises:
    """Test assert_equal_int with matching integers."""
    assert_equal_int(42, 42)


fn test_assert_equal_int_specialized_fails() raises:
    """Test assert_equal_int with mismatched integers."""
    var failed = False
    try:
        assert_equal_int(42, 43)
    except:
        failed = True
    assert_true(failed, "assert_equal_int should raise error on mismatch")


fn main() raises:
    """Run integer assertion tests."""
    test_assert_equal_int_specialized_passes()
    test_assert_equal_int_specialized_fails()
    print("All integer assertion tests passed!")
