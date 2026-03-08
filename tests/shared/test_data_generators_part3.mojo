"""Tests for data generators module - Part 3: random_normal distribution and synthetic data shapes.

# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_data_generators.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md

Tests:
- random_normal: distribution sanity
- synthetic_classification_data: shape, dtype, label values, edge cases
"""

from shared.testing import (
    random_normal,
    synthetic_classification_data,
)

# Import test helpers
from tests.shared.conftest import (
    assert_true,
    assert_dtype,
    assert_shape,
)


# ============================================================================
# Test random_normal() - distribution sanity
# ============================================================================


fn test_random_normal_distribution_sanity() raises:
    """Test random_normal produces varied values (not all same).

    With a normal distribution, consecutive values should be different.
    """
    var shape = List[Int]()
    shape.append(50)
    var tensor = random_normal(shape)

    # Get first two values
    var val1 = tensor._get_float64(0)
    var val2 = tensor._get_float64(1)

    # They should not be identical (probability of identical floats is essentially 0)
    assert_true(
        val1 != val2,
        "random_normal should produce different values for different samples",
    )


# ============================================================================
# Test synthetic_classification_data()
# ============================================================================


fn test_synthetic_classification_data_shape_features() raises:
    """Test synthetic_classification_data produces correct feature shape."""
    var result1 = synthetic_classification_data(100, 20, 3)
    var X = result1[0]

    assert_shape(X, [100, 20], "Features should have shape [100, 20]")


fn test_synthetic_classification_data_shape_labels() raises:
    """Test synthetic_classification_data produces correct label shape."""
    var result2 = synthetic_classification_data(100, 20, 3)
    var y = result2[1]

    assert_shape(y, [100], "Labels should have shape [100]")


fn test_synthetic_classification_data_dtype_features() raises:
    """Test synthetic_classification_data feature dtype."""
    var result3 = synthetic_classification_data(100, 20, 3, DType.float32)
    var X = result3[0]

    assert_dtype(X, DType.float32, "Features should have specified dtype")


fn test_synthetic_classification_data_dtype_labels() raises:
    """Test synthetic_classification_data labels are int32."""
    var result2 = synthetic_classification_data(100, 20, 3)
    var y = result2[1]

    assert_dtype(y, DType.int32, "Labels should be int32")


fn test_synthetic_classification_data_label_values() raises:
    """Test synthetic_classification_data labels are in valid range."""
    var result4 = synthetic_classification_data(50, 10, 3)
    var y = result4[1]

    # Check that labels are in [0, num_classes)
    for i in range(50):
        var label_val = y._get_int64(i)
        assert_true(label_val >= 0, "Label should be >= 0")
        assert_true(label_val < 3, "Label should be < num_classes")


fn test_synthetic_classification_data_single_sample() raises:
    """Test synthetic_classification_data with minimal size."""
    var result5 = synthetic_classification_data(1, 1, 1)
    var X = result5[0]
    var y = result5[1]

    assert_shape(X, [1, 1], "Minimal features should have shape [1, 1]")
    assert_shape(y, [1], "Minimal labels should have shape [1]")


fn test_synthetic_classification_data_many_classes() raises:
    """Test synthetic_classification_data with many classes."""
    var result6 = synthetic_classification_data(100, 10, 10)
    var y = result6[1]

    # Check all labels are in valid range
    for i in range(100):
        var label_val = y._get_int64(i)
        assert_true(label_val >= 0, "Label should be >= 0")
        assert_true(label_val < 10, "Label should be < num_classes")


fn main() raises:
    """Run random_normal distribution and synthetic classification shape tests (Part 3)."""
    print("Running random_normal distribution and synthetic data tests (Part 3)...")

    test_random_normal_distribution_sanity()
    print("✓ random_normal distribution sanity")

    test_synthetic_classification_data_shape_features()
    print("✓ synthetic_classification_data feature shape")

    test_synthetic_classification_data_shape_labels()
    print("✓ synthetic_classification_data label shape")

    test_synthetic_classification_data_dtype_features()
    print("✓ synthetic_classification_data feature dtype")

    test_synthetic_classification_data_dtype_labels()
    print("✓ synthetic_classification_data label dtype")

    test_synthetic_classification_data_label_values()
    print("✓ synthetic_classification_data label values")

    test_synthetic_classification_data_single_sample()
    print("✓ synthetic_classification_data single sample")

    test_synthetic_classification_data_many_classes()
    print("✓ synthetic_classification_data many classes")

    print("\nAll random_normal distribution and synthetic data tests passed!")
