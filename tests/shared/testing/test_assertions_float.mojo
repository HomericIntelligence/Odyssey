"""Tests for floating-point assertion functions.

Note: Split from test_assertions.mojo due to Mojo 0.26.1 heap corruption
bug that occurs after ~15 cumulative tests. See ADR-009.
"""

from testing import assert_true
from shared.testing.assertions import (
    assert_almost_equal,
    assert_dtype_equal,
    assert_equal_int,
    assert_equal_float,
    assert_close_float,
)


fn test_assert_almost_equal_float32_passes() raises:
    """Test assert_almost_equal with close Float32 values."""
    assert_almost_equal(
        Float32(1.0), Float32(1.0000001), tolerance=Float32(1e-5)
    )


fn test_assert_almost_equal_float32_fails() raises:
    """Test assert_almost_equal with distant Float32 values."""
    var failed = False
    try:
        assert_almost_equal(Float32(1.0), Float32(2.0), tolerance=Float32(1e-5))
    except:
        failed = True
    assert_true(
        failed, "assert_almost_equal should raise error for distant values"
    )


fn test_assert_almost_equal_float64_passes() raises:
    """Test assert_almost_equal with close Float64 values."""
    assert_almost_equal(
        Float64(1.0), Float64(1.0000001), tolerance=Float64(1e-5)
    )


fn test_assert_almost_equal_float64_fails() raises:
    """Test assert_almost_equal with distant Float64 values."""
    var failed = False
    try:
        assert_almost_equal(Float64(1.0), Float64(2.0), tolerance=Float64(1e-5))
    except:
        failed = True
    assert_true(
        failed, "assert_almost_equal should raise error for distant values"
    )


fn test_assert_dtype_equal_passes() raises:
    """Test assert_dtype_equal with matching dtypes."""
    assert_dtype_equal(DType.float32, DType.float32)


fn test_assert_dtype_equal_fails() raises:
    """Test assert_dtype_equal with mismatched dtypes."""
    var failed = False
    try:
        assert_dtype_equal(DType.float32, DType.float64)
    except:
        failed = True
    assert_true(
        failed, "assert_dtype_equal should raise error on mismatched dtypes"
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


fn test_assert_equal_float_passes() raises:
    """Test assert_equal_float with exactly equal floats."""
    assert_equal_float(Float32(1.0), Float32(1.0))


fn test_assert_equal_float_fails() raises:
    """Test assert_equal_float with different floats."""
    var failed = False
    try:
        assert_equal_float(Float32(1.0), Float32(1.1))
    except:
        failed = True
    assert_true(
        failed, "assert_equal_float should raise error on different values"
    )


fn main() raises:
    """Run floating-point assertion tests."""
    test_assert_almost_equal_float32_passes()
    test_assert_almost_equal_float32_fails()
    test_assert_almost_equal_float64_passes()
    test_assert_almost_equal_float64_fails()
    test_assert_dtype_equal_passes()
    test_assert_dtype_equal_fails()
    test_assert_equal_int_specialized_passes()
    test_assert_equal_int_specialized_fails()
    test_assert_equal_float_passes()
    test_assert_equal_float_fails()
    print("All floating-point assertion tests passed!")
