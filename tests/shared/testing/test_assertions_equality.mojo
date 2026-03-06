"""Tests for equality assertion functions.

Note: Split from test_assertions.mojo due to Mojo 0.26.1 heap corruption
bug that occurs after ~15 cumulative tests. See ADR-009.
"""

from testing import assert_true
from shared.testing.assertions import (
    assert_equal as custom_assert_equal,
    assert_not_equal,
    assert_not_none,
)
from collections.optional import Optional


fn test_assert_equal_int_passes() raises:
    """Test assert_equal with equal integers."""
    custom_assert_equal[Int](5, 5)


fn test_assert_equal_int_fails() raises:
    """Test assert_equal with unequal integers."""
    var failed = False
    try:
        custom_assert_equal[Int](5, 3)
    except:
        failed = True
    assert_true(failed, "assert_equal should raise error on unequal values")


fn test_assert_equal_string_passes() raises:
    """Test assert_equal with equal strings."""
    custom_assert_equal[String]("hello", "hello")


fn test_assert_equal_string_fails() raises:
    """Test assert_equal with unequal strings."""
    var failed = False
    try:
        custom_assert_equal[String]("hello", "world")
    except:
        failed = True
    assert_true(failed, "assert_equal should raise error on unequal strings")


fn test_assert_not_equal_int_passes() raises:
    """Test assert_not_equal with different integers."""
    assert_not_equal[Int](5, 3)


fn test_assert_not_equal_int_fails() raises:
    """Test assert_not_equal with equal integers."""
    var failed = False
    try:
        assert_not_equal[Int](5, 5)
    except:
        failed = True
    assert_true(failed, "assert_not_equal should raise error on equal values")


fn test_assert_not_none_passes() raises:
    """Test assert_not_none with some value."""
    var opt: Optional[Int] = Optional[Int](42)
    assert_not_none[Int](opt)


fn test_assert_not_none_fails() raises:
    """Test assert_not_none with none value."""
    var opt: Optional[Int] = Optional[Int]()
    var failed = False
    try:
        assert_not_none[Int](opt)
    except:
        failed = True
    assert_true(failed, "assert_not_none should raise error on None value")


fn main() raises:
    """Run equality assertion tests."""
    test_assert_equal_int_passes()
    test_assert_equal_int_fails()
    test_assert_equal_string_passes()
    test_assert_equal_string_fails()
    test_assert_not_equal_int_passes()
    test_assert_not_equal_int_fails()
    test_assert_not_none_passes()
    test_assert_not_none_fails()
    print("All equality assertion tests passed!")
