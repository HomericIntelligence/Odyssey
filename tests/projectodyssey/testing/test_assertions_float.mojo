"""Tests for floating-point assertion functions.

"""

from std.testing import assert_true
from projectodyssey.testing.assertions import (
    assert_almost_equal,
    assert_dtype_equal,
    assert_equal_float,
    assert_close_float,
)


def test_assert_almost_equal_float32_passes() raises:
    """Test assert_almost_equal with close Float32 values."""
    assert_almost_equal(
        Float32(1.0), Float32(1.0000001), tolerance=Float32(1e-5)
    )


def test_assert_almost_equal_float32_fails() raises:
    """Test assert_almost_equal with distant Float32 values."""
    var failed = False
    try:
        assert_almost_equal(Float32(1.0), Float32(2.0), tolerance=Float32(1e-5))
    except:
        failed = True
    assert_true(
        failed, "assert_almost_equal should raise error for distant values"
    )


def test_assert_almost_equal_float64_passes() raises:
    """Test assert_almost_equal with close Float64 values."""
    assert_almost_equal(
        Float64(1.0), Float64(1.0000001), tolerance=Float64(1e-5)
    )


def test_assert_almost_equal_float64_fails() raises:
    """Test assert_almost_equal with distant Float64 values."""
    var failed = False
    try:
        assert_almost_equal(Float64(1.0), Float64(2.0), tolerance=Float64(1e-5))
    except:
        failed = True
    assert_true(
        failed, "assert_almost_equal should raise error for distant values"
    )


def test_assert_dtype_equal_passes() raises:
    """Test assert_dtype_equal with matching dtypes."""
    assert_dtype_equal(DType.float32, DType.float32)


def test_assert_dtype_equal_fails() raises:
    """Test assert_dtype_equal with mismatched dtypes."""
    var failed = False
    try:
        assert_dtype_equal(DType.float32, DType.float64)
    except:
        failed = True
    assert_true(
        failed, "assert_dtype_equal should raise error on mismatched dtypes"
    )


def test_assert_equal_float_passes() raises:
    """Test assert_equal_float with exactly equal floats."""
    assert_equal_float(Float32(1.0), Float32(1.0))


def test_assert_equal_float_fails() raises:
    """Test assert_equal_float with different floats."""
    var failed = False
    try:
        assert_equal_float(Float32(1.0), Float32(1.1))
    except:
        failed = True
    assert_true(
        failed, "assert_equal_float should raise error on different values"
    )


def test_assert_close_float_passes() raises:
    """Test assert_close_float with values within tolerance. Closes #4096."""
    assert_close_float(1.0, 1.0001, rtol=1e-2, atol=1e-3)


def test_assert_close_float_fails() raises:
    """Test assert_close_float raises for distant values. Closes #4096."""
    var failed = False
    try:
        assert_close_float(1.0, 2.0, rtol=1e-5, atol=1e-5)
    except:
        failed = True
    assert_true(
        failed, "assert_close_float should raise error for distant values"
    )


def test_assert_close_float_passes_within_atol() raises:
    """Test assert_close_float passes when diff is within default atol."""
    # diff = 5e-9 < atol=1e-8: passes
    assert_close_float(1.0, 1.0 + 5e-9)


def test_assert_close_float_passes_within_rtol() raises:
    """Test assert_close_float passes when diff is within default rtol."""
    # diff = 5e-6, threshold = 1e-8 + 1e-5 * 1.0 = ~1e-5: passes
    assert_close_float(1.0, 1.0 + 5e-6)


def test_assert_close_float_fails_exceeds_tolerance() raises:
    """Test assert_close_float fails when diff exceeds both tolerances."""
    var failed = False
    try:
        assert_close_float(1.0, 1.5)
    except:
        failed = True
    assert_true(
        failed,
        "assert_close_float should raise error when diff exceeds tolerance",
    )


def test_assert_close_float_passes_custom_atol() raises:
    """Test assert_close_float passes with loose custom atol."""
    # diff = 0.5 < atol=1.0: passes
    assert_close_float(0.0, 0.5, atol=1.0)


def test_assert_close_float_fails_custom_atol() raises:
    """Test assert_close_float fails when diff exceeds custom atol."""
    # diff = 1.5 > atol=1.0: fails
    var failed = False
    try:
        assert_close_float(0.0, 1.5, atol=1.0, rtol=0.0)
    except:
        failed = True
    assert_true(
        failed,
        "assert_close_float should raise error when diff exceeds custom atol",
    )


def test_assert_close_float_passes_both_nan() raises:
    """Test assert_close_float passes when both values are NaN."""
    var nan = Float64(0.0) / Float64(0.0)
    assert_close_float(nan, nan)  # Both NaN: considered equal


def test_assert_close_float_fails_one_nan() raises:
    """Test assert_close_float raises when one value is NaN."""
    var nan = Float64(0.0) / Float64(0.0)
    var failed = False
    try:
        assert_close_float(1.0, nan)
    except:
        failed = True
    assert_true(
        failed, "assert_close_float should raise error for NaN mismatch"
    )


def test_assert_close_float_passes_same_inf() raises:
    """Test assert_close_float passes when both values are +inf."""
    var inf = Float64(1.0) / Float64(0.0)
    assert_close_float(inf, inf)


def test_assert_close_float_fails_inf_mismatch() raises:
    """Test assert_close_float raises when infinities have opposite sign."""
    var inf = Float64(1.0) / Float64(0.0)
    var neg_inf = Float64(-1.0) / Float64(0.0)
    var failed = False
    try:
        assert_close_float(inf, neg_inf)
    except:
        failed = True
    assert_true(
        failed, "assert_close_float should raise error for opposite infinities"
    )


# split assert_close_float tests into test_assertions_close_float.mojo.
def main() raises:
    """Run floating-point assertion tests."""
    test_assert_almost_equal_float32_passes()
    test_assert_almost_equal_float32_fails()
    test_assert_almost_equal_float64_passes()
    test_assert_almost_equal_float64_fails()
    test_assert_dtype_equal_passes()
    test_assert_dtype_equal_fails()
    test_assert_equal_float_passes()
    test_assert_equal_float_fails()
    test_assert_close_float_passes()
    test_assert_close_float_fails()
    test_assert_close_float_passes_within_atol()
    test_assert_close_float_passes_within_rtol()
    test_assert_close_float_fails_exceeds_tolerance()
    test_assert_close_float_passes_custom_atol()
    test_assert_close_float_fails_custom_atol()
    test_assert_close_float_passes_both_nan()
    test_assert_close_float_fails_one_nan()
    test_assert_close_float_passes_same_inf()
    test_assert_close_float_fails_inf_mismatch()
    print("All floating-point assertion tests passed!")
