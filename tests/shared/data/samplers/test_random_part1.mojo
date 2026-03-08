"""Tests for random sampler (part 1 of 2): creation, randomization, and correctness.

# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_random.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md

Tests RandomSampler which yields dataset indices in random order,
the standard sampling strategy for training to prevent order-dependent biases.
"""

from tests.shared.conftest import (
    assert_true,
    assert_equal,
    assert_not_equal,
    TestFixtures,
)
from shared.data.samplers import RandomSampler


# ============================================================================
# RandomSampler Creation Tests
# ============================================================================


fn test_random_sampler_creation() raises:
    """Test creating RandomSampler with dataset size.

    Should create sampler that yields all indices in random order,
    deterministic with fixed seed.
    """
    var sampler = RandomSampler(data_source_len=100)
    assert_equal(sampler.__len__(), 100)


fn test_random_sampler_with_seed() raises:
    """Test creating RandomSampler with explicit seed.

    Should accept seed parameter for deterministic shuffling,
    critical for reproducible experiments.
    """
    var sampler = RandomSampler(data_source_len=100, seed_value=42)
    assert_equal(sampler.__len__(), 100)


fn test_random_sampler_empty() raises:
    """Test creating RandomSampler with size 0.

    Should create valid sampler that yields no indices,
    edge case handling.
    """
    var sampler = RandomSampler(data_source_len=0)
    assert_equal(sampler.__len__(), 0)

    var indices = sampler.__iter__()
    assert_equal(len(indices), 0)


# ============================================================================
# RandomSampler Randomization Tests
# ============================================================================


fn test_random_sampler_shuffles_indices() raises:
    """Test that indices are shuffled, not sequential.

    Should produce different order than [0, 1, 2, ...],
    unless by extreme coincidence.
    """
    var sampler = RandomSampler(data_source_len=100, seed_value=42)
    var indices = sampler.__iter__()

    # Check that order is not sequential
    var is_sequential = True
    for i in range(100):
        if indices[i] != i:
            is_sequential = False
            break

    assert_true(not is_sequential, "Indices should be shuffled")


fn test_random_sampler_deterministic_with_seed() raises:
    """Test that same seed produces same shuffle.

    Setting seed should make shuffling deterministic,
    enabling reproducible training runs.
    """
    var sampler1 = RandomSampler(data_source_len=100, seed_value=42)
    var indices1 = sampler1.__iter__()

    var sampler2 = RandomSampler(data_source_len=100, seed_value=42)
    var indices2 = sampler2.__iter__()

    # Should produce identical shuffles
    for i in range(100):
        assert_equal(indices1[i], indices2[i])


fn test_random_sampler_varies_without_seed() raises:
    """Test that shuffle changes between epochs without fixed seed.

    Each epoch should use different random permutation,
    preventing model from learning epoch-specific patterns.
    """
    var sampler = RandomSampler(data_source_len=100)

    # First iteration
    var indices1 = sampler.__iter__()

    # Second iteration (should re-shuffle)
    var indices2 = sampler.__iter__()

    # Shuffles should likely differ (check first few indices)
    var all_same = True
    for i in range(min(10, 100)):
        if indices1[i] != indices2[i]:
            all_same = False
            break

    # It's extremely unlikely that first 10 indices match
    assert_true(not all_same, "Shuffles should differ between iterations")


# ============================================================================
# RandomSampler Correctness Tests
# ============================================================================


fn test_random_sampler_yields_all_indices() raises:
    """Test that all indices are yielded exactly once per epoch.

    Despite randomization, should yield each index [0, n-1]
    exactly once, no skipping or duplication.
    """
    var sampler = RandomSampler(data_source_len=100, seed_value=123)
    var indices = sampler.__iter__()

    # Should have all 100 indices
    assert_equal(len(indices), 100)

    # Check all indices 0-99 are present (sort to verify)
    var sorted_indices = List[Int]()
    for i in range(100):
        sorted_indices.append(indices[i])

    # Simple sort to verify all present
    for i in range(100):
        var min_idx = i
        for j in range(i + 1, 100):
            if sorted_indices[j] < sorted_indices[min_idx]:
                min_idx = j
        var temp = sorted_indices[i]
        sorted_indices[i] = sorted_indices[min_idx]
        sorted_indices[min_idx] = temp

    # After sorting, should be [0, 1, 2, ..., 99]
    for i in range(100):
        assert_equal(sorted_indices[i], i)


fn test_random_sampler_no_duplicates() raises:
    """Test that sampler doesn't yield duplicate indices.

    Each epoch should be a permutation, not sampling with replacement,
    ensuring each sample is used exactly once.
    """
    var sampler = RandomSampler(data_source_len=50, seed_value=456)
    var indices = sampler.__iter__()

    # Check for duplicates by counting occurrences
    var seen = List[Bool]()
    for _ in range(50):
        seen.append(False)

    for i in range(len(indices)):
        var idx = indices[i]
        assert_true(not seen[idx], "Index " + String(idx) + " appears twice")
        seen[idx] = True

    assert_equal(len(indices), 50)


# ============================================================================
# Main Test Runner
# ============================================================================


fn main() raises:
    """Run random sampler tests (part 1): creation, randomization, correctness."""
    print("Running random sampler tests (part 1)...")

    # Creation tests
    test_random_sampler_creation()
    test_random_sampler_with_seed()
    test_random_sampler_empty()

    # Randomization tests
    test_random_sampler_shuffles_indices()
    test_random_sampler_deterministic_with_seed()
    test_random_sampler_varies_without_seed()

    # Correctness tests
    test_random_sampler_yields_all_indices()
    test_random_sampler_no_duplicates()

    print("✓ All random sampler tests (part 1) passed!")
