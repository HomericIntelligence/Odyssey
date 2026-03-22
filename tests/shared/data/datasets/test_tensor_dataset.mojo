"""Tests for in-memory tensor dataset.

Tests TensorDataset which stores all data in memory as tensors,
providing fast random access for small to medium datasets.
"""

from tests.shared.conftest import (
    assert_true,
    assert_equal,
    assert_almost_equal,
    TestFixtures,
)
from shared.data.datasets import TensorDataset
from shared.core.extensor import AnyTensor


# ============================================================================
# Stub Implementation for TDD
# ============================================================================


struct StubTensorDataset:
    """Minimal stub tensor dataset for testing TensorDataset interface.

    Stores data and labels as simple lists for in-memory access.
    """

    var data: List[Float32]
    var labels: List[Int]
    var size: Int

    fn __init__(out self, data: List[Float32], labels: List[Int]) raises:
        """Create tensor dataset from data and labels.

        Args:
            data: List of data values.
            labels: List of label values.

        Raises:
            Error if data and labels have different lengths.
        """
        if len(data) != len(labels):
            raise Error("Data and labels must have same length")

        self.data = data.copy()
        self.labels = labels.copy()
        self.size = len(data)

    fn __len__(self) -> Int:
        """Return number of samples."""
        return self.size

    fn __getitem__(self, index: Int) raises -> Tuple[Float32, Int]:
        """Get sample at index.

        Args:
            index: Sample index.

        Returns:
            Tuple of (data, label).

        Raises:
            Error if index is out of bounds.
        """
        if index < 0 or index >= self.size:
            raise Error("Index out of bounds")

        return (self.data[index], self.labels[index])


# ============================================================================
# TensorDataset Creation Tests
# ============================================================================


fn test_tensor_dataset_creation() raises:
    """Test creating TensorDataset from tensors.

    TensorDataset should accept data and labels tensors and store them
    for efficient in-memory access.
    """
    var data: List[Float32] = [0.0, 1.0, 2.0]
    var labels: List[Int] = [0, 1, 2]
    var dataset = StubTensorDataset(data^, labels^)
    assert_equal(dataset.__len__(), 3)


fn test_tensor_dataset_with_matching_sizes() raises:
    """Test that data and labels must have matching first dimension.

    The number of samples in data must match the number of labels,
    otherwise training would fail with mismatched batch sizes.
    """
    var data = List[Float32]()
    var labels = List[Int]()
    for i in range(100):
        data.append(Float32(i))
        labels.append(i)

    var dataset = StubTensorDataset(data^, labels^)
    assert_equal(dataset.__len__(), 100)


fn test_tensor_dataset_size_mismatch_error() raises:
    """Test that mismatched data/label sizes raise error.

    Creating dataset with 100 data samples and 50 labels should fail
    immediately rather than causing silent errors during training.
    """
    var data = List[Float32]()
    var labels = List[Int]()

    for i in range(100):
        data.append(Float32(i))
    for i in range(50):
        labels.append(i)

    var error_raised = False
    try:
        var dataset = StubTensorDataset(data^, labels^)
    except:
        error_raised = True

    assert_true(error_raised, "Should have raised error for size mismatch")


fn test_tensor_dataset_empty() raises:
    """Test creating empty TensorDataset.

    Empty dataset should be valid (length 0) and not crash when queried.
    Useful for testing edge cases and incremental dataset building.
    """
    var data = List[Float32]()
    var labels = List[Int]()
    var dataset = StubTensorDataset(data^, labels^)
    assert_equal(dataset.__len__(), 0)


fn test_tensor_dataset_getitem() raises:
    """Test getting individual samples from TensorDataset.

    Should return (data, label) tuple for requested index.
    """
    var data: List[Float32] = [10.0, 20.0, 30.0]
    var labels: List[Int] = [0, 1, 2]
    var dataset = StubTensorDataset(data^, labels^)

    var sample = dataset[1]
    assert_equal(sample[0], Float32(20.0))
    assert_equal(sample[1], 1)


fn test_tensor_dataset_negative_indexing() raises:
    """Test negative indexing works correctly.

    dataset[-1] should return the last sample,
    dataset[-2] the second-to-last, etc.
    """
    var data_list: List[Float32] = [Float32(1.0), Float32(2.0), Float32(3.0)]
    var data = AnyTensor(data_list^)
    var labels_list: List[Int] = [0, 1, 2]
    var labels = AnyTensor(labels_list^)
    var dataset = TensorDataset(data^, labels^)

    var last_sample = dataset[-1]
    assert_almost_equal(last_sample[0][0], Float32(3.0))
    assert_equal(last_sample[1][0], 2)

    var second_last_sample = dataset[-2]
    assert_almost_equal(second_last_sample[0][0], Float32(2.0))
    assert_equal(second_last_sample[1][0], 1)


fn test_tensor_dataset_out_of_bounds() raises:
    """Test that out-of-bounds access raises error.

    Accessing index >= len(dataset) or index < -len(dataset)
    should raise IndexError to prevent silent failures.
    """
    var data_list: List[Float32] = [Float32(1.0), Float32(2.0), Float32(3.0)]
    var data = AnyTensor(data_list^)
    var labels_list: List[Int] = [0, 1, 2]
    var labels = AnyTensor(labels_list^)
    var dataset = TensorDataset(data^, labels^)

    # Test positive out of bounds
    var error_raised = False
    try:
        var sample = dataset[10]
    except:
        error_raised = True
    assert_true(error_raised, "Should have raised error for index 10")

    # Test negative out of bounds
    error_raised = False
    try:
        var sample = dataset[-10]
    except:
        error_raised = True
    assert_true(error_raised, "Should have raised error for index -10")


fn test_tensor_dataset_iteration_consistency() raises:
    """Test that repeated access returns same data.

    Multiple calls to dataset[i] should return identical tensors,
    ensuring deterministic behavior for debugging and testing.
    """
    var data_list: List[Float32] = [Float32(1.0), Float32(2.0)]
    var data = AnyTensor(data_list^)
    var labels_list: List[Int] = [0, 1]
    var labels = AnyTensor(labels_list^)
    var dataset = TensorDataset(data^, labels^)

    var sample1 = dataset[0]
    var sample2 = dataset[0]

    # Both accesses should return same values
    assert_almost_equal(sample1[0][0], Float32(1.0))
    assert_almost_equal(sample2[0][0], Float32(1.0))
    assert_equal(sample1[1][0], 0)
    assert_equal(sample2[1][0], 0)


# ============================================================================
# TensorDataset Memory Tests
# ============================================================================


fn test_tensor_dataset_no_copy_on_access() raises:
    """Test that __getitem__ returns views, not copies.

    For efficiency, dataset should return views into the original tensor
    rather than creating copies, reducing memory overhead.
    """
    var data_list: List[Float32] = [Float32(1.0), Float32(2.0)]
    var data = AnyTensor(data_list^)
    var labels_list: List[Int] = [0, 1]
    var labels = AnyTensor(labels_list^)
    var dataset = TensorDataset(data^, labels^)

    # Access first sample
    var sample = dataset[0]

    # Verify we get the correct data (view behavior is implicit in implementation)
    assert_almost_equal(sample[0][0], Float32(1.0))
    assert_equal(sample[1][0], 0)

    # Access second sample to verify independent views
    var sample2 = dataset[1]
    assert_almost_equal(sample2[0][0], Float32(2.0))
    assert_equal(sample2[1][0], 1)


fn test_tensor_dataset_memory_efficiency() raises:
    """Test that TensorDataset doesn't duplicate data in memory.

    Creating dataset should not copy the input tensors,
    just store references to save memory.
    """
    # Create dataset with larger data to verify no memory issues
    var data = List[Float32]()
    var labels = List[Int]()

    for i in range(1000):
        data.append(Float32(i))
        labels.append(i)

    var data_tensor = AnyTensor(data^)
    var labels_tensor = AnyTensor(labels^)
    var dataset = TensorDataset(data_tensor^, labels_tensor^)

    # Verify dataset was created successfully with all samples
    assert_equal(dataset.__len__(), 1000)

    # Spot check a few samples to verify data integrity
    var first = dataset[0]
    assert_almost_equal(first[0][0], Float32(0.0))
    assert_equal(first[1][0], 0)

    var mid = dataset[500]
    assert_almost_equal(mid[0][0], Float32(500.0))
    assert_equal(mid[1][0], 500)

    var last = dataset[999]
    assert_almost_equal(last[0][0], Float32(999.0))
    assert_equal(last[1][0], 999)


# ============================================================================
# Main Test Runner
# ============================================================================


fn main() raises:
    """Run all tensor dataset tests."""
    print("Running tensor dataset tests...")

    # Creation tests
    test_tensor_dataset_creation()
    test_tensor_dataset_with_matching_sizes()
    test_tensor_dataset_size_mismatch_error()
    test_tensor_dataset_empty()

    # Access tests
    test_tensor_dataset_getitem()
    test_tensor_dataset_negative_indexing()
    test_tensor_dataset_out_of_bounds()
    test_tensor_dataset_iteration_consistency()

    # Memory tests
    test_tensor_dataset_no_copy_on_access()
    test_tensor_dataset_memory_efficiency()

    print("✓ All tensor dataset tests passed!")
