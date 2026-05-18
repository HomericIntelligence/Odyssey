"""Unit tests for activation_ops.mojo - activation operation utilities.

Tests scalar exponential functions for numerical stability and correctness.
"""

from tests.projectodyssey.conftest import assert_almost_equal, assert_true
from projectodyssey.core.activation_ops import exp_scalar_f32, exp_scalar_f64


def test_exp_scalar_f32() raises:
    """Test exp_scalar_f32 for various inputs."""
    # Test zero: exp(0) = 1
    var result = exp_scalar_f32(0.0)
    assert_almost_equal(result, 1.0, tolerance=1e-6)

    # Test positive value: exp(1) ≈ 2.71828
    result = exp_scalar_f32(1.0)
    assert_almost_equal(result, 2.71828, tolerance=1e-4)

    # Test negative value: exp(-1) ≈ 0.36788
    result = exp_scalar_f32(-1.0)
    assert_almost_equal(result, 0.36788, tolerance=1e-4)

    # Test large negative value: exp(-20) ≈ 2e-9 (should be very small)
    result = exp_scalar_f32(-20.0)
    assert_true(result > 0.0, "exp(-20) should be positive")
    assert_true(result < 1e-8, "exp(-20) should be very small")

    # Test moderately large positive value: exp(2) ≈ 7.389
    result = exp_scalar_f32(2.0)
    assert_almost_equal(result, 7.389, tolerance=1e-2)


def test_exp_scalar_f64() raises:
    """Test exp_scalar_f64 for various inputs."""
    # Test zero: exp(0) = 1
    var result = exp_scalar_f64(0.0)
    assert_almost_equal(Float32(result), 1.0, tolerance=1e-6)

    # Test positive value: exp(1) ≈ 2.71828182845904523536
    result = exp_scalar_f64(1.0)
    assert_almost_equal(Float32(result), 2.71828, tolerance=1e-4)

    # Test negative value: exp(-1) ≈ 0.36787944117144232160
    result = exp_scalar_f64(-1.0)
    assert_almost_equal(Float32(result), 0.36788, tolerance=1e-4)

    # Test large negative value: exp(-20)
    result = exp_scalar_f64(-20.0)
    assert_true(result > 0.0, "exp(-20) should be positive")
    assert_true(result < 1e-8, "exp(-20) should be very small")

    # Test moderately large positive value: exp(2) ≈ 7.389
    result = exp_scalar_f64(2.0)
    assert_almost_equal(Float32(result), 7.389, tolerance=1e-2)


def test_exp_scalar_f32_monotonicity() raises:
    """Test that exp is monotonically increasing."""
    # exp(a) < exp(b) for a < b
    var exp_neg_1 = exp_scalar_f32(-1.0)
    var exp_0 = exp_scalar_f32(0.0)
    var exp_1 = exp_scalar_f32(1.0)

    assert_true(exp_neg_1 < exp_0, "exp(-1) should be less than exp(0)")
    assert_true(exp_0 < exp_1, "exp(0) should be less than exp(1)")


def test_exp_scalar_f64_monotonicity() raises:
    """Test that exp is monotonically increasing for float64."""
    var exp_neg_1 = exp_scalar_f64(-1.0)
    var exp_0 = exp_scalar_f64(0.0)
    var exp_1 = exp_scalar_f64(1.0)

    assert_true(exp_neg_1 < exp_0, "exp(-1) should be less than exp(0)")
    assert_true(exp_0 < exp_1, "exp(0) should be less than exp(1)")


def main() raises:
    """Run all tests."""
    test_exp_scalar_f32()
    test_exp_scalar_f64()
    test_exp_scalar_f32_monotonicity()
    test_exp_scalar_f64_monotonicity()
    print("All activation_ops tests passed!")
