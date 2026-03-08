"""Tests for weighted sampler - Part 2: Class Balancing, Determinism, and Error Tests.

# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_weighted.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md

Tests WeightedSampler which samples indices according to specified weights,
enabling class balancing and importance sampling for imbalanced datasets.
"""

from tests.shared.conftest import (
    assert_true,
    assert_equal,
    assert_almost_equal,
    assert_greater,
    TestFixtures,
)
from shared.data.samplers import WeightedSampler


# ============================================================================
# WeightedSampler Class Balancing Tests
# ============================================================================


fn test_weighted_sampler_class_balancing() raises:
    """Test using WeightedSampler for class balancing.

    For imbalanced dataset (90 class A, 10 class B),
    weights can balance classes to 50/50 sampling.
    """
    # Simulated dataset: 90 samples of class 0, 10 of class 1
    var weights = List[Float64]()
    for _ in range(90):
        weights.append(1.0 / 90.0)  # Low weight for majority class
    for _ in range(10):
        weights.append(1.0 / 10.0)  # High weight for minority class

    var sampler = WeightedSampler(weights^, num_samples=1000, seed_value=555)
    var indices = sampler.__iter__()

    var count_class0 = 0
    var count_class1 = 0
    for i in range(len(indices)):
        if indices[i] < 90:
            count_class0 += 1
        else:
            count_class1 += 1

    # Should be approximately balanced (50/50)
    assert_true(count_class0 > 450 and count_class0 < 550)
    assert_true(count_class1 > 450 and count_class1 < 550)


fn test_weighted_sampler_inverse_frequency() raises:
    """Test inverse frequency weighting for balancing.

    Weight = 1/class_frequency is common pattern for balancing,
    e.g., class with 100 samples gets weight 1/100.
    """
    # Class frequencies: [100, 50, 10]
    var weights = List[Float64]()
    for _ in range(100):
        weights.append(1.0 / 100.0)
    for _ in range(50):
        weights.append(1.0 / 50.0)
    for _ in range(10):
        weights.append(1.0 / 10.0)

    var sampler = WeightedSampler(weights^, num_samples=1500, seed_value=666)
    var indices = sampler.__iter__()

    var counts = List[Int]()
    counts.append(0)
    counts.append(0)
    counts.append(0)

    for i in range(len(indices)):
        var idx = indices[i]
        if idx < 100:
            counts[0] += 1
        elif idx < 150:
            counts[1] += 1
        else:
            counts[2] += 1

    # All classes should be approximately balanced
    for i in range(3):
        assert_true(counts[i] > 450 and counts[i] < 550)


# ============================================================================
# WeightedSampler Determinism Tests
# ============================================================================


fn test_weighted_sampler_deterministic_with_seed() raises:
    """Test that weighted sampling is deterministic with seed.

    Same seed should produce same sequence of indices,
    enabling reproducible training.
    """
    var weights1: List[Float64] = []
    weights1.append(1.0)
    weights1.append(2.0)
    weights1.append(3.0)
    var sampler1 = WeightedSampler(weights1^, num_samples=100, seed_value=777)
    var indices1 = sampler1.__iter__()

    var weights2: List[Float64] = []
    weights2.append(1.0)
    weights2.append(2.0)
    weights2.append(3.0)
    var sampler2 = WeightedSampler(weights2^, num_samples=100, seed_value=777)
    var indices2 = sampler2.__iter__()

    # Should produce same sequence with same seed
    for i in range(100):
        assert_equal(indices1[i], indices2[i])


# ============================================================================
# WeightedSampler Multiple Iterations Tests
# ============================================================================


fn test_weighted_sampler_multiple_iterations() raises:
    """Verify WeightedSampler can be iterated multiple times.

    When called multiple times, should produce valid samples each time,
    not get stuck or fail on second iteration.
    """
    var weights: List[Float64] = []
    weights.append(0.1)
    weights.append(0.2)
    weights.append(0.3)
    weights.append(0.4)
    var sampler = WeightedSampler(weights^, num_samples=4, seed_value=888)

    # First iteration
    var indices1 = sampler.__iter__()
    assert_equal(len(indices1), 4)

    # Second iteration should work the same
    var indices2 = sampler.__iter__()
    assert_equal(len(indices2), 4)


# ============================================================================
# WeightedSampler Error Handling Tests
# ============================================================================


fn test_weighted_sampler_negative_weight_error() raises:
    """Test that negative weights raise error.

    Weights must be non-negative (>=0),
    negative weights are meaningless for probabilities.
    """
    var weights: List[Float64] = []
    weights.append(1.0)
    weights.append(-1.0)
    weights.append(2.0)

    var error_raised = False
    try:
        var sampler = WeightedSampler(weights^, num_samples=100)
    except:
        error_raised = True

    assert_true(error_raised, "Should have raised error for negative weight")


fn test_weighted_sampler_all_zero_weights_error() raises:
    """Test that all-zero weights raise error.

    If all weights are zero, cannot sample anything,
    should fail with clear error.
    """
    var weights: List[Float64] = []
    weights.append(0.0)
    weights.append(0.0)
    weights.append(0.0)

    var error_raised = False
    try:
        var sampler = WeightedSampler(weights^, num_samples=100)
    except:
        error_raised = True

    assert_true(error_raised, "Should have raised error for all-zero weights")


fn test_weighted_sampler_empty_weights_error() raises:
    """Test that empty weights list raises error.

    Cannot create sampler with no weights,
    should fail immediately.
    """
    var weights: List[Float64] = []

    var error_raised = False
    try:
        var sampler = WeightedSampler(weights^, num_samples=100)
    except:
        error_raised = True

    assert_true(error_raised, "Should have raised error for empty weights")


# ============================================================================
# Main Test Runner
# ============================================================================


fn main() raises:
    """Run weighted sampler part 2 tests."""
    print("Running weighted sampler part 2 tests...")

    # Class balancing tests
    test_weighted_sampler_class_balancing()
    test_weighted_sampler_inverse_frequency()

    # Determinism tests
    test_weighted_sampler_deterministic_with_seed()

    # Multiple iterations tests
    test_weighted_sampler_multiple_iterations()

    # Error handling tests
    test_weighted_sampler_negative_weight_error()
    test_weighted_sampler_all_zero_weights_error()
    test_weighted_sampler_empty_weights_error()

    print("✓ All weighted sampler part 2 tests passed!")
