"""Tests for weighted sampler

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


fn test_weighted_sampler_creation() raises:
    """Test creating WeightedSampler with weights.

    Should accept list of weights (one per sample) and sample
    indices proportional to weights.
    """
    var weights: List[Float64] = []
    weights.append(1.0)
    weights.append(2.0)
    weights.append(3.0)
    weights.append(4.0)
    var sampler = WeightedSampler(weights^, num_samples=100, seed_value=42)
    assert_equal(sampler.__len__(), 100)


fn test_weighted_sampler_uniform_weights() raises:
    """Test that uniform weights produce uniform sampling.

    When all weights are equal, should behave like RandomSampler,
    each index equally likely.
    """
    var weights: List[Float64] = []
    weights.append(1.0)
    weights.append(1.0)
    weights.append(1.0)
    weights.append(1.0)
    var sampler = WeightedSampler(weights^, num_samples=1000, seed_value=123)
    var indices = sampler.__iter__()

    var counts = List[Int]()
    counts.append(0)
    counts.append(0)
    counts.append(0)
    counts.append(0)

    for i in range(len(indices)):
        counts[indices[i]] += 1

    # Each index should appear ~250 times (25% of 1000)
    for i in range(4):
        assert_true(counts[i] > 200 and counts[i] < 300)


fn test_weighted_sampler_zero_weight() raises:
    """Test that zero-weight samples are (almost) never sampled.

    Indices with weight=0 should (theoretically) never be yielded,
    useful for excluding certain samples.
    Note: Due to normalization, this tests very low weights.
    """
    var weights: List[Float64] = []
    weights.append(1.0)
    weights.append(0.001)  # Very low weight
    weights.append(1.0)
    weights.append(0.001)  # Very low weight
    var sampler = WeightedSampler(weights^, num_samples=100, seed_value=456)
    var indices = sampler.__iter__()

    var counts = List[Int]()
    counts.append(0)
    counts.append(0)
    counts.append(0)
    counts.append(0)

    for i in range(len(indices)):
        counts[indices[i]] += 1

    # Indices 0 and 2 should dominate, 1 and 3 should be rare
    assert_true(counts[0] > 40)
    assert_true(counts[2] > 40)


fn test_weighted_sampler_weights_normalization() raises:
    """Test that weights are automatically normalized.

    Weights [1, 2, 3] should behave same as [10, 20, 30],
    only relative proportions matter.
    """
    var weights1: List[Float64] = []
    weights1.append(1.0)
    weights1.append(2.0)
    weights1.append(3.0)
    var sampler1 = WeightedSampler(weights1^, num_samples=100, seed_value=789)
    var indices1 = sampler1.__iter__()

    var weights2: List[Float64] = []
    weights2.append(10.0)
    weights2.append(20.0)
    weights2.append(30.0)
    var sampler2 = WeightedSampler(weights2^, num_samples=100, seed_value=789)
    var indices2 = sampler2.__iter__()

    # Should produce same sequence (weights are proportionally equivalent)
    for i in range(100):
        assert_equal(indices1[i], indices2[i])


fn test_weighted_sampler_proportional_sampling() raises:
    """Test that sampling is proportional to weights.

    With weights [1, 2, 3], index 2 should appear 3x more often
    than index 0, index 1 appears 2x more often than index 0.
    """
    var weights: List[Float64] = []
    weights.append(1.0)
    weights.append(2.0)
    weights.append(3.0)
    var sampler = WeightedSampler(weights^, num_samples=6000, seed_value=111)
    var indices = sampler.__iter__()

    var counts = List[Int]()
    counts.append(0)
    counts.append(0)
    counts.append(0)

    for i in range(len(indices)):
        counts[indices[i]] += 1

    # Expected proportions: 1/6, 2/6, 3/6
    # With 6000 samples: ~1000, ~2000, ~3000
    assert_true(counts[0] > 800 and counts[0] < 1200)  # ~1000
    assert_true(counts[1] > 1800 and counts[1] < 2200)  # ~2000
    assert_true(counts[2] > 2800 and counts[2] < 3200)  # ~3000


fn test_weighted_sampler_extreme_weights() raises:
    """Test handling of extreme weight ratios.

    With weights [0.001, 0.999], second index should dominate
    (appear ~99.9% of the time).
    """
    var weights: List[Float64] = []
    weights.append(0.001)
    weights.append(0.999)
    var sampler = WeightedSampler(weights^, num_samples=1000, seed_value=222)
    var indices = sampler.__iter__()

    var count_idx1 = 0
    for i in range(len(indices)):
        if indices[i] == 1:
            count_idx1 += 1

    # Should be approximately 999 out of 1000 (99.9%)
    assert_true(count_idx1 > 990)


fn test_weighted_sampler_with_replacement() raises:
    """Test that weighted sampling uses replacement by default.

    Should allow same index to be sampled multiple times,
    as sampling is probabilistic.
    """
    var weights: List[Float64] = []
    weights.append(1.0)
    weights.append(1.0)
    var sampler = WeightedSampler(
        weights^, num_samples=100, replacement=True, seed_value=333
    )
    var indices = sampler.__iter__()

    # With replacement, we expect duplicates
    assert_equal(len(indices), 100)


fn test_weighted_sampler_num_samples() raises:
    """Test that num_samples controls total samples yielded.

    Should yield exactly num_samples indices,
    regardless of dataset size or weights.
    """
    var weights: List[Float64] = []
    weights.append(1.0)
    weights.append(1.0)
    weights.append(1.0)
    var sampler = WeightedSampler(weights^, num_samples=1000, seed_value=444)
    var indices = sampler.__iter__()

    assert_equal(len(indices), 1000)


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


fn main() raises:
    """Run all test_weighted tests."""
    print("Running test_weighted tests...")

    test_weighted_sampler_creation()
    print("✓ test_weighted_sampler_creation")

    test_weighted_sampler_uniform_weights()
    print("✓ test_weighted_sampler_uniform_weights")

    test_weighted_sampler_zero_weight()
    print("✓ test_weighted_sampler_zero_weight")

    test_weighted_sampler_weights_normalization()
    print("✓ test_weighted_sampler_weights_normalization")

    test_weighted_sampler_proportional_sampling()
    print("✓ test_weighted_sampler_proportional_sampling")

    test_weighted_sampler_extreme_weights()
    print("✓ test_weighted_sampler_extreme_weights")

    test_weighted_sampler_with_replacement()
    print("✓ test_weighted_sampler_with_replacement")

    test_weighted_sampler_num_samples()
    print("✓ test_weighted_sampler_num_samples")

    test_weighted_sampler_class_balancing()
    print("✓ test_weighted_sampler_class_balancing")

    test_weighted_sampler_inverse_frequency()
    print("✓ test_weighted_sampler_inverse_frequency")

    test_weighted_sampler_deterministic_with_seed()
    print("✓ test_weighted_sampler_deterministic_with_seed")

    test_weighted_sampler_multiple_iterations()
    print("✓ test_weighted_sampler_multiple_iterations")

    test_weighted_sampler_negative_weight_error()
    print("✓ test_weighted_sampler_negative_weight_error")

    test_weighted_sampler_all_zero_weights_error()
    print("✓ test_weighted_sampler_all_zero_weights_error")

    test_weighted_sampler_empty_weights_error()
    print("✓ test_weighted_sampler_empty_weights_error")

    print("\nAll test_weighted tests passed!")
