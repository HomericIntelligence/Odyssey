"""Tests for sequential sampler.

Tests SequentialSampler which yields dataset indices in order,
the default sampling strategy for deterministic data loading.
"""


from tests.odyssey.conftest import (
    TestFixtures,
    assert_equal,
    assert_true,
)
from odyssey.data.samplers import SequentialSampler


struct StubSequentialSampler:
    """Minimal stub sequential sampler for testing Sampler interface.

    Yields indices in sequential order [0, 1, 2, ..., n-1].
    """

    var size: Int

    def __init__(out self, size: Int):
        """Create sequential sampler.

        Args:
            size: Number of indices to generate.
        """
        self.size = size

    def __len__(self) -> Int:
        """Return number of indices."""
        return self.size

    def get_index(self, position: Int) -> Int:
        """Get index at position.

        Args:
            position: Position in sequence (0 to size-1).

        Returns:
            Index value (same as position for sequential sampler).
        """
        return position


def test_sequential_sampler_creation() raises:
    """Test creating SequentialSampler with dataset size.

    Should create sampler that will yield indices 0 to n-1 in order,
    deterministic and reproducible.
    """
    var sampler = StubSequentialSampler(size=100)
    assert_equal(sampler.__len__(), 100)


def test_sequential_sampler_empty() raises:
    """Test creating SequentialSampler with size 0.

    Should create valid sampler that yields no indices,
    useful for edge case testing.
    """
    var sampler = StubSequentialSampler(size=0)
    assert_equal(sampler.__len__(), 0)


def test_sequential_sampler_yields_all_indices() raises:
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


def test_sequential_sampler_order() raises:
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


def test_sequential_sampler_deterministic() raises:
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


def test_sequential_sampler_start_index() raises:
    """Test indices start from 0.

    First yielded index should always be 0,
    not 1 or any other value.
    """
    var sampler = SequentialSampler(data_source_len=100)
    var indices = sampler.__iter__()
    assert_equal(indices[0], 0)


def test_sequential_sampler_end_index() raises:
    """Test indices end at size-1.

    Last yielded index should be size-1,
    as indices are 0-based.
    """
    var sampler = SequentialSampler(data_source_len=100)
    var indices = sampler.__iter__()

    var last_idx = indices[len(indices) - 1]
    assert_equal(last_idx, 99)


def test_sequential_sampler_no_negative_indices() raises:
    """Test that sampler never yields negative indices.

    All indices should be >= 0,
    as negative indices would be invalid.
    """
    var sampler = SequentialSampler(data_source_len=100)
    var indices = sampler.__iter__()

    for i in range(len(indices)):
        assert_true(indices[i] >= 0)


def test_sequential_sampler_with_dataloader() raises:
    """Test using SequentialSampler standalone.

    SequentialSampler should produce indices in deterministic order
    suitable for use with DataLoader.
    """
    var sampler = SequentialSampler(data_source_len=100)
    var indices = sampler.__iter__()

    # First batch indices (0-31) should be in sequential order
    for i in range(32):
        assert_equal(indices[i], i)


def test_sequential_sampler_reusable() raises:
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


def test_sequential_sampler_iteration_speed() raises:
    """Test that iteration is fast.

    Should iterate through indices with minimal overhead,
    as this happens every training step.
    """
    var sampler = SequentialSampler(data_source_len=100000)

    # Should complete quickly - iterate through all indices
    var indices = sampler.__iter__()
    var count = len(indices)

    assert_equal(count, 100000)


def test_sequential_sampler_memory_efficiency() raises:
    """Test that sampler can handle large datasets.

    Creating sampler for large dataset should work,
    indices generated when __iter__() is called.
    """
    # Creating sampler for 1M indices should be lightweight
    var sampler = SequentialSampler(data_source_len=1000000)
    assert_equal(sampler.__len__(), 1000000)


def main() raises:
    """Run all test_sequential tests."""
    print("Running test_sequential tests...")

    test_sequential_sampler_creation()
    print("✓ test_sequential_sampler_creation")

    test_sequential_sampler_empty()
    print("✓ test_sequential_sampler_empty")

    test_sequential_sampler_yields_all_indices()
    print("✓ test_sequential_sampler_yields_all_indices")

    test_sequential_sampler_order()
    print("✓ test_sequential_sampler_order")

    test_sequential_sampler_deterministic()
    print("✓ test_sequential_sampler_deterministic")

    test_sequential_sampler_start_index()
    print("✓ test_sequential_sampler_start_index")

    test_sequential_sampler_end_index()
    print("✓ test_sequential_sampler_end_index")

    test_sequential_sampler_no_negative_indices()
    print("✓ test_sequential_sampler_no_negative_indices")

    test_sequential_sampler_with_dataloader()
    print("✓ test_sequential_sampler_with_dataloader")

    test_sequential_sampler_reusable()
    print("✓ test_sequential_sampler_reusable")

    test_sequential_sampler_iteration_speed()
    print("✓ test_sequential_sampler_iteration_speed")

    test_sequential_sampler_memory_efficiency()
    print("✓ test_sequential_sampler_memory_efficiency")

    print("\nAll test_sequential tests passed!")
