"""Tests for gradient checking utilities in projectodyssey.testing.gradient_checker.

Tests relative_error and assert_gradients_close functions.
Gradient checking with check_gradients is tested in test_gradient_checking.mojo.
"""

from std.testing import assert_true, assert_equal
from projectodyssey.testing.gradient_checker import (
    assert_gradients_close,
    relative_error,
)
from projectodyssey.tensor.any_tensor import AnyTensor, zeros, ones, full


# ============================================================================
# Relative Error Tests
# ============================================================================


def test_relative_error_identical() raises:
    """Test relative_error with identical values.

    Expected: relative_error(x, x) = 0.0
    """
    print("Testing relative_error with identical values...")
    var err = relative_error(1.0, 1.0)
    assert_true(err < 1e-10, "Relative error should be ~0")
    print("  ✓ Relative error identical values correct")


def test_relative_error_small_difference() raises:
    """Test relative_error with small difference.

    Expected: relative_error(1.0, 1.001) ≈ 0.001
    """
    print("Testing relative_error with small difference...")
    var err = relative_error(1.0, 1.001)
    assert_true(err > 0.0009 and err < 0.0011, "Relative error mismatch")
    print("  ✓ Relative error small difference correct")


def test_relative_error_near_zero() raises:
    """Test relative_error with values near zero.

    Formula: |a - n| / max(|a|, |n|, 1e-8)
    Should not divide by zero when both values are near zero.
    """
    print("Testing relative_error near zero values...")
    var err = relative_error(1e-9, 1e-9)
    assert_true(
        err < 1e-10, "Relative error should handle near-zero gracefully"
    )
    print("  ✓ Relative error near-zero handling correct")


def test_relative_error_large_difference() raises:
    """Test relative_error with large difference."""
    print("Testing relative_error with large difference...")
    var err = relative_error(1.0, 2.0)
    # |1-2| / max(1, 2, 1e-8) = 1/2 = 0.5
    assert_true(err > 0.49 and err < 0.51, "Relative error should be ~0.5")
    print("  ✓ Relative error large difference correct")


# ============================================================================
# Gradient Assertion Tests
# ============================================================================


def test_assert_gradients_close_identical() raises:
    """Test assert_gradients_close with identical gradients."""
    print("Testing assert_gradients_close with identical gradients...")
    var shape = [3, 4]
    var grad1 = full(shape, 1.5, DType.float32)
    var grad2 = full(shape, 1.5, DType.float32)

    # Should not raise
    assert_gradients_close(grad1, grad2)
    print("  ✓ assert_gradients_close identical gradients passes")


def test_assert_gradients_close_within_tolerance() raises:
    """Test assert_gradients_close with close gradients."""
    print("Testing assert_gradients_close within tolerance...")
    var shape = [3, 4]
    var grad1 = full(shape, 1.0, DType.float32)
    var grad2 = full(shape, 1.001, DType.float32)

    # Should not raise with default tolerances
    assert_gradients_close(grad1, grad2, rtol=1e-2, atol=1e-2)
    print("  ✓ assert_gradients_close within tolerance passes")


def test_assert_gradients_close_mismatched_shape() raises:
    """Test assert_gradients_close with mismatched shapes."""
    print("Testing assert_gradients_close with mismatched shapes...")
    var shape1 = [3, 4]
    var shape2 = [4, 5]
    var grad1 = full(shape1, 1.0, DType.float32)
    var grad2 = full(shape2, 1.0, DType.float32)

    var failed = False
    try:
        assert_gradients_close(grad1, grad2)
    except:
        failed = True

    assert_true(failed, "Should raise error on shape mismatch")
    print("  ✓ assert_gradients_close shape mismatch detection works")


def test_assert_gradients_close_mismatched_dtype() raises:
    """Test assert_gradients_close with mismatched dtypes."""
    print("Testing assert_gradients_close with mismatched dtypes...")
    var shape = [3, 4]
    var grad1 = full(shape, 1.0, DType.float32)
    var grad2 = full(shape, 1.0, DType.float64)

    var failed = False
    try:
        assert_gradients_close(grad1, grad2)
    except:
        failed = True

    assert_true(failed, "Should raise error on dtype mismatch")
    print("  ✓ assert_gradients_close dtype mismatch detection works")


def test_assert_gradients_close_zero_gradients() raises:
    """Test assert_gradients_close with zero gradients."""
    print("Testing assert_gradients_close with zero gradients...")
    var shape = [2, 3]
    var grad1 = zeros(shape, DType.float32)
    var grad2 = zeros(shape, DType.float32)

    # Should not raise
    assert_gradients_close(grad1, grad2)
    print("  ✓ assert_gradients_close handles zero gradients")


# ============================================================================
# Main Test Runner
# ============================================================================


def main() raises:
    """Run all gradient checking tests."""
    print("=" * 60)
    print("GRADIENT CHECKING UTILITY TESTS")
    print("=" * 60)

    # Relative error tests
    test_relative_error_identical()
    test_relative_error_small_difference()
    test_relative_error_near_zero()
    test_relative_error_large_difference()

    # Gradient assertions
    test_assert_gradients_close_identical()
    test_assert_gradients_close_within_tolerance()
    test_assert_gradients_close_mismatched_shape()
    test_assert_gradients_close_mismatched_dtype()
    test_assert_gradients_close_zero_gradients()

    print("=" * 60)
    print("All gradient checking utility tests passed!")
    print("=" * 60)
