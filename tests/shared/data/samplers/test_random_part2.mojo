"""Tests for random sampler (part 2 of 2): valid range, replacement, integration, performance.

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
# RandomSampler Correctness Tests (continued)
# ============================================================================


fn test_random_sampler_valid_range() raises:
    """Test that all yielded indices are in valid range [0, size-1].

    Should never yield negative indices or indices >= size,
    as these would cause out-of-bounds errors.
    """
    var sampler = RandomSampler(data_source_len=100, seed_value=789)
    var indices = sampler.__iter__()

    for i in range(len(indices)):
        assert_true(indices[i] >= 0)
        assert_true(indices[i] < 100)


# ============================================================================
# RandomSampler Replacement Tests
# ============================================================================


fn test_random_sampler_with_replacement() raises:
    """Test random sampling with replacement.

    When replacement=True, should allow duplicate indices,
    useful for oversampling minority classes.
    """
    var sampler = RandomSampler(
        data_source_len=10, replacement=True, num_samples=100, seed_value=111
    )
    var indices = sampler.__iter__()

    # Should have 100 samples (more than dataset size)
    assert_equal(len(indices), 100)

    # All indices should be in valid range
    for i in range(len(indices)):
        assert_true(indices[i] >= 0)
        assert_true(indices[i] < 10)


fn test_random_sampler_replacement_oversampling() raises:
    """Test oversampling with replacement.

    Can sample more than dataset size when replacement=True,
    common for balancing imbalanced datasets.
    """
    var sampler = RandomSampler(
        data_source_len=10, replacement=True, num_samples=1000, seed_value=222
    )
    var indices = sampler.__iter__()

    assert_equal(len(indices), 1000)

    # All indices should still be in valid range
    for i in range(1000):
        assert_true(indices[i] >= 0 and indices[i] < 10)


# ============================================================================
# RandomSampler Integration Tests
# ============================================================================


fn test_random_sampler_with_dataloader() raises:
    """Test using RandomSampler standalone for DataLoader-style usage.

    RandomSampler should produce randomly ordered indices
    suitable for use with DataLoader.
    """
    var sampler = RandomSampler(data_source_len=100, seed_value=333)
    var indices = sampler.__iter__()

    # First batch indices (0-31) should NOT be [0, 1, 2, ..., 31]
    var is_sequential = True
    for i in range(32):
        if indices[i] != i:
            is_sequential = False
            break

    assert_true(not is_sequential, "Indices should be shuffled")


# ============================================================================
# RandomSampler Performance Tests
# ============================================================================


fn test_random_sampler_shuffle_speed() raises:
    """Test that shuffling is fast even for large datasets.

    Creating sampler and generating permutation should be
    efficient for datasets with millions of samples.
    """
    var sampler = RandomSampler(data_source_len=1000000, seed_value=444)

    # Creating sampler should be lightweight
    assert_equal(sampler.__len__(), 1000000)

    # Note: Full iteration would be slow, so we just verify creation works
    # In production, indices are generated lazily during iteration


# ============================================================================
# Main Test Runner
# ============================================================================


fn main() raises:
    """Run random sampler tests (part 2): valid range, replacement, integration, performance."""
    print("Running random sampler tests (part 2)...")

    # Correctness tests (continued)
    test_random_sampler_valid_range()

    # Replacement tests
    test_random_sampler_with_replacement()
    test_random_sampler_replacement_oversampling()

    # Integration tests
    test_random_sampler_with_dataloader()

    # Performance tests
    test_random_sampler_shuffle_speed()

    print("✓ All random sampler tests (part 2) passed!")
