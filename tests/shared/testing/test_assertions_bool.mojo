"""Tests for boolean assertion functions.

Note: Split from test_assertions.mojo due to Mojo 0.26.1 heap corruption
bug that occurs after ~15 cumulative tests. See ADR-009.
"""

from testing import assert_true
from shared.testing.assertions import (
    assert_true as custom_assert_true,
    assert_false,
)


fn test_assert_true_passes() raises:
    """Test assert_true with true condition."""
    custom_assert_true(True)


fn test_assert_true_fails() raises:
    """Test assert_true with false condition."""
    var failed = False
    try:
        custom_assert_true(False)
    except:
        failed = True
    assert_true(failed, "assert_true should raise error on false condition")


fn test_assert_true_custom_message() raises:
    """Test assert_true with custom error message."""
    var failed = False
    var caught_message = False
    try:
        custom_assert_true(False, "Custom error message")
    except e:
        failed = True
        var msg = String(e)
        caught_message = "Custom error message" in msg
    assert_true(failed, "assert_true should raise error")
    assert_true(caught_message, "Error message should contain custom text")


fn test_assert_false_passes() raises:
    """Test assert_false with false condition."""
    assert_false(False)


fn test_assert_false_fails() raises:
    """Test assert_false with true condition."""
    var failed = False
    try:
        assert_false(True)
    except:
        failed = True
    assert_true(failed, "assert_false should raise error on true condition")


fn main() raises:
    """Run boolean assertion tests."""
    test_assert_true_passes()
    test_assert_true_fails()
    test_assert_true_custom_message()
    test_assert_false_passes()
    test_assert_false_fails()
    print("All boolean assertion tests passed!")
