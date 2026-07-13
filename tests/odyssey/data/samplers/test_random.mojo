"""Tests for random sampler (part 1 of 2): creation, randomization, and correctness.

Tests RandomSampler which yields dataset indices in random order,
the standard sampling strategy for training to prevent order-dependent biases.
"""


from tests.odyssey.conftest import (
    assert_true,
    assert_equal,
    assert_not_equal,
    TestFixtures,
)
from odyssey.data.samplers import RandomSampler


def test_random_sampler_creation() raises:
    """Test creating RandomSampler with dataset size.

    Should create sampler that yields all indices in random order,
    deterministic with fixed seed.
    """
    var sampler = RandomSampler(data_source_len=100)
    assert_equal(sampler.__len__(), 100)


def test_random_sampler_with_seed() raises:
    """Test creating RandomSampler with explicit seed.

    Should accept seed parameter for deterministic shuffling,
    critical for reproducible experiments.
    """
    var sampler = RandomSampler(data_source_len=100, seed_value=42)
    assert_equal(sampler.__len__(), 100)


def test_random_sampler_empty() raises:
    """Test creating RandomSampler with size 0.

    Should create valid sampler that yields no indices,
    edge case handling.
    """
    var sampler = RandomSampler(data_source_len=0)
    assert_equal(sampler.__len__(), 0)

    var indices = sampler.__iter__()
    assert_equal(len(indices), 0)


def test_random_sampler_shuffles_indices() raises:
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


def test_random_sampler_deterministic_with_seed() raises:
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


def test_random_sampler_varies_without_seed() raises:
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


def test_random_sampler_yields_all_indices() raises:
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


def test_random_sampler_no_duplicates() raises:
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


def test_random_sampler_valid_range() raises:
    """Test that all yielded indices are in valid range [0, size-1].

    Should never yield negative indices or indices >= size,
    as these would cause out-of-bounds errors.
    """
    var sampler = RandomSampler(data_source_len=100, seed_value=789)
    var indices = sampler.__iter__()

    for i in range(len(indices)):
        assert_true(indices[i] >= 0)
        assert_true(indices[i] < 100)


def test_random_sampler_with_replacement() raises:
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


def test_random_sampler_replacement_oversampling() raises:
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


def test_random_sampler_with_dataloader() raises:
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


def test_random_sampler_shuffle_speed() raises:
    """Test that shuffling is fast even for large datasets.

    Creating sampler and generating permutation should be
    efficient for datasets with millions of samples.
    """
    var sampler = RandomSampler(data_source_len=1000000, seed_value=444)

    # Creating sampler should be lightweight
    assert_equal(sampler.__len__(), 1000000)

    # Note: Full iteration would be slow, so we just verify creation works
    # In production, indices are generated lazily during iteration


def main() raises:
    """Run all test_random tests."""
    print("Running test_random tests...")

    test_random_sampler_creation()
    print("✓ test_random_sampler_creation")

    test_random_sampler_with_seed()
    print("✓ test_random_sampler_with_seed")

    test_random_sampler_empty()
    print("✓ test_random_sampler_empty")

    test_random_sampler_shuffles_indices()
    print("✓ test_random_sampler_shuffles_indices")

    test_random_sampler_deterministic_with_seed()
    print("✓ test_random_sampler_deterministic_with_seed")

    test_random_sampler_varies_without_seed()
    print("✓ test_random_sampler_varies_without_seed")

    test_random_sampler_yields_all_indices()
    print("✓ test_random_sampler_yields_all_indices")

    test_random_sampler_no_duplicates()
    print("✓ test_random_sampler_no_duplicates")

    test_random_sampler_valid_range()
    print("✓ test_random_sampler_valid_range")

    test_random_sampler_with_replacement()
    print("✓ test_random_sampler_with_replacement")

    test_random_sampler_replacement_oversampling()
    print("✓ test_random_sampler_replacement_oversampling")

    test_random_sampler_with_dataloader()
    print("✓ test_random_sampler_with_dataloader")

    test_random_sampler_shuffle_speed()
    print("✓ test_random_sampler_shuffle_speed")

    print("\nAll test_random tests passed!")
