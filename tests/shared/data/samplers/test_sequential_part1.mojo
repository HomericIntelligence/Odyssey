# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_sequential.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""Tests for sequential sampler - Part 1: Creation, iteration, and range tests.

Tests SequentialSampler which yields dataset indices in order,
the default sampling strategy for deterministic data loading.
"""

from tests.shared.conftest import assert_true, assert_equal, TestFixtures
from shared.data.samplers import SequentialSampler


# ============================================================================
# Stub Implementation for TDD
# ============================================================================


struct StubSequentialSampler:
    """Minimal stub sequential sampler for testing Sampler interface.

    Yields indices in sequential order [0, 1, 2, ..., n-1].
    """

    var size: Int

    fn __init__(out self, size: Int):
        """Create sequential sampler.

        Args:
            size: Number of indices to generate.
        """
        self.size = size

    fn __len__(self) -> Int:
        """Return number of indices."""
        return self.size

    fn get_index(self, position: Int) -> Int:
        """Get index at position.

        Args:
            position: Position in sequence (0 to size-1).

        Returns:
            Index value (same as position for sequential sampler).
        """
        return position


# ============================================================================
# SequentialSampler Creation Tests
# ============================================================================


fn test_sequential_sampler_creation() raises:
    """Test creating SequentialSampler with dataset size.

    Should create sampler that will yield indices 0 to n-1 in order,
    deterministic and reproducible.
    """
    var sampler = StubSequentialSampler(size=100)
    assert_equal(sampler.__len__(), 100)


fn test_sequential_sampler_empty() raises:
    """Test creating SequentialSampler with size 0.

    Should create valid sampler that yields no indices,
    useful for edge case testing.
    """
    var sampler = StubSequentialSampler(size=0)
    assert_equal(sampler.__len__(), 0)


# ============================================================================
# SequentialSampler Iteration Tests
# ============================================================================


fn test_sequential_sampler_yields_all_indices() raises:
    """Test that sampler yields all indices exactly once.

    Should produce indices [0, 1, 2, ..., n-1] without
    skipping or duplicating any.
    """
    var sampler = StubSequentialSampler(size=10)

    var indices = List[Int]()
    for i in range(sampler.__len__()):
        indices.append(sampler.get_index(i))

    assert_equal(len(indices), 10)

    # Check all indices present and in order
    for i in range(10):
        assert_equal(indices[i], i)


fn test_sequential_sampler_order() raises:
    """Test that indices are yielded in sequential order.

    Should yield [0, 1, 2, 3, ...], not shuffled or reversed.
    This is the defining property of SequentialSampler.
    """
    var sampler = StubSequentialSampler(size=100)

    var indices = List[Int]()
    for i in range(sampler.__len__()):
        indices.append(sampler.get_index(i))

    # Check indices are in order
    for i in range(100):
        assert_equal(indices[i], i)


fn test_sequential_sampler_deterministic() raises:
    """Test that sampler produces same sequence every time.

    Multiple iterations should yield identical index sequences,
    no randomness involved.
    """
    var sampler = StubSequentialSampler(size=50)

    # First iteration
    var indices1 = List[Int]()
    for i in range(sampler.__len__()):
        indices1.append(sampler.get_index(i))

    # Second iteration
    var indices2 = List[Int]()
    for i in range(sampler.__len__()):
        indices2.append(sampler.get_index(i))

    # Should be identical
    for i in range(50):
        assert_equal(indices1[i], indices2[i])


# ============================================================================
# SequentialSampler Range Tests
# ============================================================================


fn test_sequential_sampler_start_index() raises:
    """Test indices start from 0.

    First yielded index should always be 0,
    not 1 or any other value.
    """
    var sampler = SequentialSampler(data_source_len=100)
    var indices = sampler.__iter__()
    assert_equal(indices[0], 0)


fn test_sequential_sampler_end_index() raises:
    """Test indices end at size-1.

    Last yielded index should be size-1,
    as indices are 0-based.
    """
    var sampler = SequentialSampler(data_source_len=100)
    var indices = sampler.__iter__()

    var last_idx = indices[len(indices) - 1]
    assert_equal(last_idx, 99)


fn test_sequential_sampler_no_negative_indices() raises:
    """Test that sampler never yields negative indices.

    All indices should be >= 0,
    as negative indices would be invalid.
    """
    var sampler = SequentialSampler(data_source_len=100)
    var indices = sampler.__iter__()

    for i in range(len(indices)):
        assert_true(indices[i] >= 0)


# ============================================================================
# Main Test Runner
# ============================================================================


fn main() raises:
    """Run sequential sampler tests - Part 1."""
    print("Running sequential sampler tests (part 1)...")

    # Creation tests
    test_sequential_sampler_creation()
    test_sequential_sampler_empty()

    # Iteration tests
    test_sequential_sampler_yields_all_indices()
    test_sequential_sampler_order()
    test_sequential_sampler_deterministic()

    # Range tests
    test_sequential_sampler_start_index()
    test_sequential_sampler_end_index()
    test_sequential_sampler_no_negative_indices()

    print("✓ All sequential sampler tests (part 1) passed!")
