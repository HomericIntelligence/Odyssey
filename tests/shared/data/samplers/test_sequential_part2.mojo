# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_sequential.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""Tests for sequential sampler - Part 2: Integration and performance tests.

Tests SequentialSampler which yields dataset indices in order,
the default sampling strategy for deterministic data loading.
"""

from tests.shared.conftest import assert_equal
from shared.data.samplers import SequentialSampler


# ============================================================================
# SequentialSampler Integration Tests
# ============================================================================


fn test_sequential_sampler_with_dataloader() raises:
    """Test using SequentialSampler standalone.

    SequentialSampler should produce indices in deterministic order
    suitable for use with DataLoader.
    """
    var sampler = SequentialSampler(data_source_len=100)
    var indices = sampler.__iter__()

    # First batch indices (0-31) should be in sequential order
    for i in range(32):
        assert_equal(indices[i], i)


fn test_sequential_sampler_reusable() raises:
    """Test that sampler can be reused across multiple epochs.

    Same sampler instance should work for multiple epochs,
    yielding same sequence each time.
    """
    var sampler = SequentialSampler(data_source_len=50)

    # Epoch 1
    var epoch1_indices = sampler.__iter__()

    # Epoch 2 (reuse same sampler)
    var epoch2_indices = sampler.__iter__()

    # Should produce identical sequences
    assert_equal(len(epoch1_indices), len(epoch2_indices))
    for i in range(50):
        assert_equal(epoch1_indices[i], epoch2_indices[i])


# ============================================================================
# SequentialSampler Performance Tests
# ============================================================================


fn test_sequential_sampler_iteration_speed() raises:
    """Test that iteration is fast.

    Should iterate through indices with minimal overhead,
    as this happens every training step.
    """
    var sampler = SequentialSampler(data_source_len=100000)

    # Should complete quickly - iterate through all indices
    var indices = sampler.__iter__()
    var count = len(indices)

    assert_equal(count, 100000)


fn test_sequential_sampler_memory_efficiency() raises:
    """Test that sampler can handle large datasets.

    Creating sampler for large dataset should work,
    indices generated when __iter__() is called.
    """
    # Creating sampler for 1M indices should be lightweight
    var sampler = SequentialSampler(data_source_len=1000000)
    assert_equal(sampler.__len__(), 1000000)


# ============================================================================
# Main Test Runner
# ============================================================================


fn main() raises:
    """Run sequential sampler tests - Part 2."""
    print("Running sequential sampler tests (part 2)...")

    # Integration tests
    test_sequential_sampler_with_dataloader()
    test_sequential_sampler_reusable()

    # Performance tests
    test_sequential_sampler_iteration_speed()
    test_sequential_sampler_memory_efficiency()

    print("✓ All sequential sampler tests (part 2) passed!")
