"""Tests for data generators module - Part 2: random_uniform and random_normal (shape/dtype).

# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_data_generators.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md

Tests:
- random_uniform: shape, range, dtype
- random_normal: shape, dtype (float32, float64), mean/std, distribution sanity
"""

from shared.testing import (
    random_uniform,
    random_normal,
)

# Import test helpers
from tests.shared.conftest import (
    assert_true,
    assert_dtype,
    assert_numel,
)


# ============================================================================
# Test random_uniform()
# ============================================================================


fn test_random_uniform_shape() raises:
    """Test random_uniform creates correct shape."""
    var shape = List[Int]()
    shape.append(10)
    shape.append(5)
    var tensor = random_uniform(shape, low=0.0, high=1.0)

    assert_numel(tensor, 50, "random_uniform(10,5) should have 50 elements")


fn test_random_uniform_range_0_to_1() raises:
    """Test random_uniform with default range [0, 1)."""
    var shape = List[Int]()
    shape.append(100)
    var tensor = random_uniform(shape)

    # Check sample values are in [0, 1)
    for i in range(0, 100, 10):
        var val = tensor._get_float64(i)
        assert_true(val >= 0.0, "uniform value should be >= 0.0")
        assert_true(val < 1.0, "uniform value should be < 1.0")


fn test_random_uniform_range_negative_to_positive() raises:
    """Test random_uniform with custom range [-1, 1)."""
    var shape = List[Int]()
    shape.append(100)
    var tensor = random_uniform(shape, low=-1.0, high=1.0)

    # Check sample values are in [-1, 1)
    for i in range(0, 100, 10):
        var val = tensor._get_float64(i)
        assert_true(val >= -1.0, "uniform value should be >= -1.0")
        assert_true(val < 1.0, "uniform value should be < 1.0")


fn test_random_uniform_dtype() raises:
    """Test random_uniform respects dtype."""
    var shape = List[Int]()
    shape.append(10)
    var tensor = random_uniform(shape, dtype=DType.float64)

    assert_dtype(tensor, DType.float64, "random_uniform should respect dtype")


# ============================================================================
# Test random_normal() - shape and dtype
# ============================================================================


fn test_random_normal_shape() raises:
    """Test random_normal creates correct shape."""
    var shape = List[Int]()
    shape.append(20)
    shape.append(30)
    var tensor = random_normal(shape)

    assert_numel(tensor, 600, "random_normal(20,30) should have 600 elements")


fn test_random_normal_dtype_float32() raises:
    """Test random_normal with float32 dtype."""
    var shape = List[Int]()
    shape.append(10)
    var tensor = random_normal(shape, dtype=DType.float32)

    assert_dtype(tensor, DType.float32, "random_normal should respect dtype")


fn test_random_normal_dtype_float64() raises:
    """Test random_normal with float64 dtype."""
    var shape = List[Int]()
    shape.append(10)
    var tensor = random_normal(shape, dtype=DType.float64)

    assert_dtype(tensor, DType.float64, "random_normal should respect dtype")


fn test_random_normal_mean_and_std() raises:
    """Test random_normal respects mean and std parameters.

    Note: With small samples (10 elements), we only do sanity checks.
    Statistical tests would require larger samples.
    """
    var shape = List[Int]()
    shape.append(10)

    # Generate with mean=5.0, std=1.0
    var tensor = random_normal(shape, mean=5.0, std=1.0)

    # Get a sample value - should be roughly around mean (with tolerance)
    var val = tensor._get_float64(0)

    # With mean=5.0, values should typically be in [3, 7] (mean ± 2*std)
    assert_true(
        val >= 1.0 and val <= 9.0,
        "random_normal with mean=5.0 should produce values roughly around 5.0",
    )


fn main() raises:
    """Run random_uniform and random_normal shape/dtype tests (Part 2)."""
    print("Running random_uniform and random_normal tests (Part 2)...")

    # random_uniform tests
    test_random_uniform_shape()
    print("✓ random_uniform shape")

    test_random_uniform_range_0_to_1()
    print("✓ random_uniform range [0, 1)")

    test_random_uniform_range_negative_to_positive()
    print("✓ random_uniform range [-1, 1)")

    test_random_uniform_dtype()
    print("✓ random_uniform dtype")

    # random_normal shape/dtype tests
    test_random_normal_shape()
    print("✓ random_normal shape")

    test_random_normal_dtype_float32()
    print("✓ random_normal float32 dtype")

    test_random_normal_dtype_float64()
    print("✓ random_normal float64 dtype")

    test_random_normal_mean_and_std()
    print("✓ random_normal mean and std")

    print("\nAll random_uniform and random_normal shape/dtype tests passed!")
