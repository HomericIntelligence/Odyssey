"""Tests for base dataset interface.

Tests the abstract Dataset interface that all datasets must implement,
ensuring consistent API across different dataset types.
"""

from tests.shared.conftest import assert_true, assert_equal, TestFixtures


# ============================================================================
# Stub Implementation for TDD
# ============================================================================


struct StubDataset:
    """Minimal stub dataset for testing Dataset interface requirements.

    This stub provides the minimum interface that all datasets must implement,
    allowing tests to execute and validate the Dataset API contract.
    """

    var size: Int
    var data: List[Float32]

    def __init__(out self, size: Int):
        """Create stub dataset with specified size.

        Args:
            size: Number of samples in the dataset.
        """
        self.size = size
        self.data = List[Float32]()
        for i in range(size):
            self.data.append(Float32(i))

    def __len__(self) -> Int:
        """Return number of samples in dataset."""
        return self.size

    def __getitem__(self, index: Int) raises -> Tuple[Float32, Int]:
        """Get sample at index.

        Args:
            index: Sample index (supports negative indexing).

        Returns:
            Tuple of (data, label).

        Raises:
            Error if index is out of bounds.
        """
        var actual_index = index
        if index < 0:
            actual_index = self.size + index

        if actual_index < 0 or actual_index >= self.size:
            raise Error("Index out of bounds")

        return (self.data[actual_index], actual_index)


# ============================================================================
# Base Dataset Interface Tests
# ============================================================================


def test_dataset_has_len_method() raises:
    """Test that Dataset interface requires __len__ method.

    The Dataset trait must provide a way to query the total number of samples.
    This is critical for batch calculation and progress tracking.
    """
    var dataset = StubDataset(size=100)
    assert_equal(dataset.__len__(), 100)


def test_dataset_has_getitem_method() raises:
    """Test that Dataset interface requires __getitem__ method.

    The Dataset trait must provide indexed access to samples.
    This enables data loaders to fetch specific samples by index.
    """
    var dataset = StubDataset(size=100)
    var sample = dataset[0]
    # Verify sample is a valid tuple
    assert_equal(sample[1], 0)


def test_dataset_getitem_returns_tuple() raises:
    """Test that __getitem__ returns (data, label) tuple.

    Standard convention is to return both data and label together,
    which simplifies training loop implementations.
    """
    var dataset = StubDataset(size=100)
    var data_label = dataset[0]
    var data = data_label[0]
    var label = data_label[1]
    assert_equal(data, Float32(0))
    assert_equal(label, 0)


def test_dataset_getitem_index_validation() raises:
    """Test that __getitem__ validates index bounds.

    Should raise error for out-of-bounds indices to prevent
    accessing invalid memory or returning corrupted data.
    """
    var dataset = StubDataset(size=100)
    var error_raised = False
    try:
        var _ = dataset[100]  # Out of bounds
    except:
        error_raised = True
    assert_true(
        error_raised, "Should have raised error for out of bounds index"
    )


def test_dataset_supports_negative_indexing() raises:
    """Test that Dataset supports Python-style negative indexing.

    Negative indices should count from the end: dataset[-1] == dataset[len-1].
    This provides convenient access to the last elements.
    """
    var dataset = StubDataset(size=100)
    var last_sample = dataset[-1]
    var explicit_last = dataset[99]
    assert_equal(last_sample[0], explicit_last[0])
    assert_equal(last_sample[1], explicit_last[1])


def test_dataset_length_immutable() raises:
    """Test that dataset length remains constant after creation.

    The __len__ method should return the same value across multiple calls,
    ensuring deterministic behavior for data loaders.
    """
    var dataset = StubDataset(size=100)
    var len1 = dataset.__len__()
    var len2 = dataset.__len__()
    assert_equal(len1, len2)


def test_dataset_iteration_consistency() raises:
    """Test that repeated __getitem__ calls return consistent data.

    Calling dataset[i] multiple times should return the same data,
    unless the dataset explicitly supports randomization per access.
    """
    var dataset = StubDataset(size=100)
    var sample1 = dataset[5]
    var sample2 = dataset[5]
    assert_equal(sample1[0], sample2[0])
    assert_equal(sample1[1], sample2[1])


# ============================================================================
# Main Test Runner
# ============================================================================


def main() raises:
    """Run all base dataset tests."""
    print("Running base dataset tests...")

    test_dataset_has_len_method()
    test_dataset_has_getitem_method()
    test_dataset_getitem_returns_tuple()
    test_dataset_getitem_index_validation()
    test_dataset_supports_negative_indexing()
    test_dataset_length_immutable()
    test_dataset_iteration_consistency()

    print("✓ All base dataset tests passed!")
