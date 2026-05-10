"""Tests for GRADIENT_CHECK_EPSILON_FLOAT32 and GRADIENT_CHECK_EPSILON_OTHER constants.

Verifies that both constants are importable from the submodule and the package,
values are consistent between both import paths, and numeric values are correct.
"""

from std.testing import assert_true, assert_equal
from shared.testing.tolerance_constants import (
    GRADIENT_CHECK_EPSILON,
    GRADIENT_CHECK_EPSILON_FLOAT32,
    GRADIENT_CHECK_EPSILON_FLOAT32 as EPSILON_FLOAT32_DIRECT,
    GRADIENT_CHECK_EPSILON_OTHER,
    GRADIENT_CHECK_EPSILON_OTHER as EPSILON_OTHER_DIRECT,
)

# ============================================================================
# Value Correctness Tests
# ============================================================================


def test_gradient_check_epsilon_float32_value() raises:
    """Test that GRADIENT_CHECK_EPSILON_FLOAT32 has the correct value (3e-4)."""
    print("Testing GRADIENT_CHECK_EPSILON_FLOAT32 value...")
    assert_true(
        GRADIENT_CHECK_EPSILON_FLOAT32 == 3e-4,
        "GRADIENT_CHECK_EPSILON_FLOAT32 should be 3e-4",
    )
    print("  GRADIENT_CHECK_EPSILON_FLOAT32 == 3e-4 correct")


def test_gradient_check_epsilon_other_value() raises:
    """Test that GRADIENT_CHECK_EPSILON_OTHER has the correct value (1e-3)."""
    print("Testing GRADIENT_CHECK_EPSILON_OTHER value...")
    assert_true(
        GRADIENT_CHECK_EPSILON_OTHER == 1e-3,
        "GRADIENT_CHECK_EPSILON_OTHER should be 1e-3",
    )
    print("  GRADIENT_CHECK_EPSILON_OTHER == 1e-3 correct")


# ============================================================================
# Import Path Consistency Tests
# ============================================================================


def test_epsilon_float32_package_equals_submodule() raises:
    """Test that package import and submodule import give identical values."""
    print("Testing GRADIENT_CHECK_EPSILON_FLOAT32 package == submodule...")
    assert_true(
        GRADIENT_CHECK_EPSILON_FLOAT32 == EPSILON_FLOAT32_DIRECT,
        (
            "Package and submodule values for GRADIENT_CHECK_EPSILON_FLOAT32"
            " must match"
        ),
    )
    print("  Package and submodule values match")


def test_epsilon_other_package_equals_submodule() raises:
    """Test that package import and submodule import give identical values."""
    print("Testing GRADIENT_CHECK_EPSILON_OTHER package == submodule...")
    assert_true(
        GRADIENT_CHECK_EPSILON_OTHER == EPSILON_OTHER_DIRECT,
        (
            "Package and submodule values for GRADIENT_CHECK_EPSILON_OTHER must"
            " match"
        ),
    )
    print("  Package and submodule values match")


# ============================================================================
# Ordering / Relationship Tests
# ============================================================================


def test_epsilon_float32_larger_than_generic() raises:
    """Test that GRADIENT_CHECK_EPSILON_FLOAT32 > GRADIENT_CHECK_EPSILON.

    Float32 matmul precision loss requires a larger epsilon than the generic 1e-5.
    """
    print("Testing GRADIENT_CHECK_EPSILON_FLOAT32 > GRADIENT_CHECK_EPSILON...")
    assert_true(
        GRADIENT_CHECK_EPSILON_FLOAT32 > GRADIENT_CHECK_EPSILON,
        (
            "GRADIENT_CHECK_EPSILON_FLOAT32 must be larger than"
            " GRADIENT_CHECK_EPSILON"
        ),
    )
    print("  GRADIENT_CHECK_EPSILON_FLOAT32 > GRADIENT_CHECK_EPSILON correct")


def test_epsilon_other_larger_than_float32() raises:
    """Test that GRADIENT_CHECK_EPSILON_OTHER > GRADIENT_CHECK_EPSILON_FLOAT32.
    """
    print(
        "Testing GRADIENT_CHECK_EPSILON_OTHER >"
        " GRADIENT_CHECK_EPSILON_FLOAT32..."
    )
    assert_true(
        GRADIENT_CHECK_EPSILON_OTHER > GRADIENT_CHECK_EPSILON_FLOAT32,
        (
            "GRADIENT_CHECK_EPSILON_OTHER must be larger than"
            " GRADIENT_CHECK_EPSILON_FLOAT32"
        ),
    )
    print(
        "  GRADIENT_CHECK_EPSILON_OTHER > GRADIENT_CHECK_EPSILON_FLOAT32"
        " correct"
    )


# ============================================================================
# Main Test Runner
# ============================================================================


def main() raises:
    """Run all gradient epsilon constant tests."""
    print("=" * 60)
    print("GRADIENT EPSILON CONSTANTS TESTS")
    print("=" * 60)

    test_gradient_check_epsilon_float32_value()
    test_gradient_check_epsilon_other_value()
    test_epsilon_float32_package_equals_submodule()
    test_epsilon_other_package_equals_submodule()
    test_epsilon_float32_larger_than_generic()
    test_epsilon_other_larger_than_float32()

    print("=" * 60)
    print("All gradient epsilon constant tests passed!")
    print("=" * 60)
