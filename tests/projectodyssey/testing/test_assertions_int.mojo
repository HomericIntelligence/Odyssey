"""Tests for integer assertion functions.

# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under

def test_ target per file.
"""

from std.testing import assert_true
from projectodyssey.testing.assertions import (
    assert_equal_int,
)


def test_assert_equal_int_specialized_passes() raises:
    """Test assert_equal_int with matching integers."""
    assert_equal_int(42, 42)


def test_assert_equal_int_specialized_fails() raises:
    """Test assert_equal_int with mismatched integers."""
    var failed = False
    try:
        assert_equal_int(42, 43)
    except:
        failed = True
    assert_true(failed, "assert_equal_int should raise error on mismatch")


def main() raises:
    """Run integer assertion tests."""
    test_assert_equal_int_specialized_passes()
    test_assert_equal_int_specialized_fails()
    print("All integer assertion tests passed!")
