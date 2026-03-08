"""Tests for weighted sampler - Part 1: Creation, Probability, and Replacement Tests.

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
# WeightedSampler Creation Tests
# ============================================================================


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


# ============================================================================
# WeightedSampler Probability Tests
# ============================================================================


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


# ============================================================================
# WeightedSampler Replacement Tests
# ============================================================================


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


# ============================================================================
# Main Test Runner
# ============================================================================


fn main() raises:
    """Run weighted sampler part 1 tests."""
    print("Running weighted sampler part 1 tests...")

    # Creation tests
    test_weighted_sampler_creation()
    test_weighted_sampler_uniform_weights()
    test_weighted_sampler_zero_weight()
    test_weighted_sampler_weights_normalization()

    # Probability tests
    test_weighted_sampler_proportional_sampling()
    test_weighted_sampler_extreme_weights()

    # Replacement tests
    test_weighted_sampler_with_replacement()
    test_weighted_sampler_num_samples()

    print("✓ All weighted sampler part 1 tests passed!")
