"""Tests for comparison assertion functions.

Note: Split from test_assertions.mojo due to Mojo 0.26.1 heap corruption
bug that occurs after ~15 cumulative tests. See ADR-009.
"""

from testing import assert_true
from shared.testing.assertions import (
    assert_close_float,
    assert_greater,
    assert_less,
    assert_greater_or_equal,
    assert_less_or_equal,
)


fn test_assert_close_float_passes() raises:
    """Test assert_close_float with numerically close values."""
    assert_close_float(1.0, 1.00001, rtol=1e-3, atol=1e-3)


fn test_assert_close_float_fails() raises:
    """Test assert_close_float with distant values."""
    var failed = False
    try:
        assert_close_float(1.0, 10.0, rtol=1e-3, atol=1e-3)
    except:
        failed = True
    assert_true(
        failed, "assert_close_float should raise error for distant values"
    )


fn test_assert_greater_float32_passes() raises:
    """Test assert_greater with a > b (Float32)."""
    assert_greater(Float32(2.0), Float32(1.0))


fn test_assert_greater_float32_fails() raises:
    """Test assert_greater with a <= b (Float32)."""
    var failed = False
    try:
        assert_greater(Float32(1.0), Float32(2.0))
    except:
        failed = True
    assert_true(failed, "assert_greater should raise error when a <= b")


fn test_assert_greater_int_passes() raises:
    """Test assert_greater with a > b (Int)."""
    assert_greater(2, 1)


fn test_assert_greater_int_fails() raises:
    """Test assert_greater with a <= b (Int)."""
    var failed = False
    try:
        assert_greater(1, 2)
    except:
        failed = True
    assert_true(failed, "assert_greater should raise error when a <= b")


fn test_assert_less_float32_passes() raises:
    """Test assert_less with a < b (Float32)."""
    assert_less(Float32(1.0), Float32(2.0))


fn test_assert_less_float32_fails() raises:
    """Test assert_less with a >= b (Float32)."""
    var failed = False
    try:
        assert_less(Float32(2.0), Float32(1.0))
    except:
        failed = True
    assert_true(failed, "assert_less should raise error when a >= b")


fn main() raises:
    """Run comparison assertion tests."""
    test_assert_close_float_passes()
    test_assert_close_float_fails()
    test_assert_greater_float32_passes()
    test_assert_greater_float32_fails()
    test_assert_greater_int_passes()
    test_assert_greater_int_fails()
    test_assert_less_float32_passes()
    test_assert_less_float32_fails()
    print("All comparison assertion tests passed!")
