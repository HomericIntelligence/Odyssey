"""Tests for data generators module - Part 4: high-dimensional data and integration tests.

# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_data_generators.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md

Tests:
- synthetic_classification_data: high-dimensional features
- Integration: combining random_tensor and random_normal
- Integration: verifying classification data shape consistency
"""

from shared.testing import (
    random_tensor,
    random_normal,
    synthetic_classification_data,
)

# Import test helpers
from tests.shared.conftest import (
    assert_equal_int,
    assert_shape,
    assert_numel,
)


# ============================================================================
# Test synthetic_classification_data() - high dimensions
# ============================================================================


fn test_synthetic_classification_data_high_dimensions() raises:
    """Test synthetic_classification_data with high-dimensional features."""
    var result7 = synthetic_classification_data(50, 100, 5)
    var X = result7[0]

    assert_shape(X, [50, 100], "Should handle high-dimensional features")


# ============================================================================
# Integration Tests
# ============================================================================


fn test_integration_random_tensor_and_normal() raises:
    """Test combining random tensor and normal generation."""
    var shape = List[Int]()
    shape.append(5)
    shape.append(5)

    var t1 = random_tensor(shape)
    var t2 = random_normal(shape)

    assert_numel(t1, 25, "First tensor should have 25 elements")
    assert_numel(t2, 25, "Second tensor should have 25 elements")


fn test_integration_classification_data_shapes_match() raises:
    """Test features and labels shapes are consistent."""
    var num_samples = 100
    var result8 = synthetic_classification_data(num_samples, 15, 4)
    var X = result8[0]
    var y = result8[1]

    # Number of samples should match
    assert_equal_int(
        X.shape()[0],
        num_samples,
        "Features first dimension should match num_samples",
    )
    assert_equal_int(
        y.shape()[0], num_samples, "Labels should match num_samples"
    )


fn main() raises:
    """Run high-dimensional and integration tests (Part 4)."""
    print("Running high-dimensional and integration tests (Part 4)...")

    test_synthetic_classification_data_high_dimensions()
    print("✓ synthetic_classification_data high dimensions")

    test_integration_random_tensor_and_normal()
    print("✓ integration random tensor and normal")

    test_integration_classification_data_shapes_match()
    print("✓ integration classification data shapes match")

    print("\nAll high-dimensional and integration tests passed!")
