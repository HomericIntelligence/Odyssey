"""Tests for base data loader interface.

Tests the DataLoader interface that all loaders must implement,
ensuring consistent API for batch iteration during training.
"""

from tests.shared.conftest import assert_true, assert_equal, TestFixtures


# ============================================================================
# Stub Implementations for TDD
# ============================================================================


struct StubDataset:
    """Minimal stub dataset for loader testing."""

    var size: Int

    def __init__(out self, size: Int):
        self.size = size

    def __len__(self) -> Int:
        return self.size

    def __getitem__(self, index: Int) raises -> Tuple[Float32, Int]:
        if index < 0 or index >= self.size:
            raise Error("Index out of bounds")
        return (Float32(index), index)


struct StubBatch(Copyable, Movable, Sized):
    """Minimal stub batch for testing batch operations."""

    var batch_size: Int

    def __init__(out self, capacity: Int):
        self.batch_size = capacity

    def __moveinit__(out self, deinit existing: Self):
        """Move initializer."""
        self.batch_size = existing.batch_size

    def __len__(self) -> Int:
        """Return batch size for len() builtin."""
        return self.batch_size

    def size(self) -> Int:
        return self.batch_size


struct StubDataLoader:
    """Minimal stub data loader for testing DataLoader interface.

    Provides basic batching functionality without complex features.
    """

    var dataset_size: Int
    var batch_size: Int
    var drop_last: Bool
    var num_batches: Int

    def __init__(
        out self,
        dataset: StubDataset,
        batch_size: Int,
        drop_last: Bool = False,
    ) raises:
        """Create stub data loader.

        Args:
            dataset: Dataset to load from.
            batch_size: Number of samples per batch.
            drop_last: Whether to drop last incomplete batch.

        Raises:
            Error if batch_size <= 0.
        """
        if batch_size <= 0:
            raise Error("batch_size must be positive")

        self.dataset_size = dataset.__len__()
        self.batch_size = batch_size
        self.drop_last = drop_last

        # Calculate number of batches
        var n_samples = self.dataset_size
        if n_samples == 0:
            self.num_batches = 0
        elif drop_last:
            self.num_batches = n_samples // batch_size
        else:
            self.num_batches = (n_samples + batch_size - 1) // batch_size

    def __len__(self) -> Int:
        """Return number of batches."""
        return self.num_batches

    def get_batch(self, batch_idx: Int) raises -> StubBatch:
        """Get batch at specified index.

        Args:
            batch_idx: Batch index.

        Returns:
            Batch containing samples.
        """
        var start_idx = batch_idx * self.batch_size
        var end_idx = start_idx + self.batch_size

        if end_idx > self.dataset_size:
            end_idx = self.dataset_size

        var num_samples = end_idx - start_idx
        var batch = StubBatch(capacity=num_samples)
        return batch^


# ============================================================================
# Base DataLoader Interface Tests
# ============================================================================


def test_loader_has_len_method() raises:
    """Test that DataLoader interface requires __len__ method.

    Should return the number of batches (not samples),
    enabling progress bars and epoch calculations.
    """
    var dataset = StubDataset(size=100)
    var loader = StubDataLoader(dataset, batch_size=32)
    assert_equal(loader.__len__(), 4)  # ceil(100/32) = 4 batches


def test_loader_batch_size_consistency() raises:
    """Test that batches have consistent size (except possibly last).

    All batches should have batch_size samples except the last batch
    which may be smaller if dataset size is not divisible by batch_size.
    """
    var dataset = StubDataset(size=100)
    var loader = StubDataLoader(dataset, batch_size=32)

    # First 3 batches should be size 32
    var batch0 = loader.get_batch(0)
    assert_equal(len(batch0), 32)

    var batch1 = loader.get_batch(1)
    assert_equal(len(batch1), 32)

    var batch2 = loader.get_batch(2)
    assert_equal(len(batch2), 32)

    # Last batch should be size 4 (100 - 3*32)
    var batch3 = loader.get_batch(3)
    assert_equal(len(batch3), 4)


def test_loader_empty_dataset() raises:
    """Test loader behavior with empty dataset.

    Should create valid loader that yields zero batches,
    not crash when iterated.
    """
    var dataset = StubDataset(size=0)
    var loader = StubDataLoader(dataset, batch_size=32)
    assert_equal(loader.__len__(), 0)


def test_loader_single_sample() raises:
    """Test loader with single sample dataset.

    Should create one batch containing the single sample,
    even though batch_size is larger.
    """
    var dataset = StubDataset(size=1)
    var loader = StubDataLoader(dataset, batch_size=32)

    assert_equal(loader.__len__(), 1)

    var batch = loader.get_batch(0)
    assert_equal(len(batch), 1)


# ============================================================================
# DataLoader Configuration Tests
# ============================================================================


def test_loader_batch_size_validation() raises:
    """Test that batch_size must be positive.

    Creating loader with batch_size <= 0 should raise ValueError,
    as it would cause division by zero or infinite loop.
    """
    var dataset = StubDataset(size=100)
    var error_raised = False
    try:
        var loader = StubDataLoader(dataset, batch_size=0)
    except:
        error_raised = True
    assert_true(error_raised, "Should have raised error for batch_size <= 0")


def test_loader_drop_last_option() raises:
    """Test drop_last parameter.

    When drop_last=True, should discard final partial batch,
    ensuring all batches have exactly batch_size samples.
    """
    var dataset = StubDataset(size=100)
    var loader = StubDataLoader(dataset, batch_size=32, drop_last=True)

    # Should drop last batch of 4 samples
    assert_equal(loader.__len__(), 3)

    # All remaining batches should have exactly batch_size samples
    for i in range(loader.__len__()):
        var batch = loader.get_batch(i)
        assert_equal(len(batch), 32)


# ============================================================================
# Main Test Runner
# ============================================================================


def main() raises:
    """Run all base loader tests."""
    print("Running base loader tests...")

    # Interface tests
    test_loader_has_len_method()
    test_loader_batch_size_consistency()
    test_loader_empty_dataset()
    test_loader_single_sample()

    # Configuration tests
    test_loader_batch_size_validation()
    test_loader_drop_last_option()

    print("✓ All base loader tests passed!")
