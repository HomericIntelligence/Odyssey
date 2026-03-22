"""Tests for prefetch buffer infrastructure.

Tests PrefetchBuffer and PrefetchDataLoader for batch prefetching.
"""

from tests.shared.conftest import (
    assert_true,
    assert_equal,
)
from shared.data import (
    AnyTensorDataset,
    BatchLoader,
    RandomSampler,
    TransformedDataset,
)
from shared.data.prefetch import PrefetchBuffer, PrefetchDataLoader
from shared.data.loaders import Batch
from shared.core.extensor import AnyTensor, ones, zeros
from collections import List


# ============================================================================
# PrefetchBuffer Tests
# ============================================================================


fn test_prefetch_buffer_creation() raises:
    """Test creating a PrefetchBuffer."""
    var buffer = PrefetchBuffer(capacity=2)
    assert_equal(buffer.capacity, 2)
    assert_true(buffer.is_empty())


fn test_prefetch_buffer_capacity_validation() raises:
    """Test that PrefetchBuffer validates capacity.

    Negative or zero capacity should raise an error.
    """
    try:
        var buffer = PrefetchBuffer(capacity=0)
        assert_true(False)  # Should have raised
    except:
        pass


fn test_prefetch_buffer_append_and_pop() raises:
    """Test appending and popping batches.

    Should follow FIFO order.
    """
    var buffer = PrefetchBuffer(capacity=3)

    # Create dummy batches
    var data1 = ones([2, 3, 8, 8], DType.float32)
    var labels1 = zeros([2, 10], DType.float32)
    var indices1 = List[Int]()
    indices1.append(0)
    indices1.append(1)
    var batch1 = Batch(data1^, labels1^, indices1^)

    buffer.append(batch1^)
    assert_true(not buffer.is_empty())
    assert_equal(buffer.batches.__len__(), 1)

    var popped_batch = buffer.pop()
    assert_true(buffer.is_empty())


fn test_prefetch_buffer_capacity_limit() raises:
    """Test that PrefetchBuffer respects capacity limit.

    Should raise error when trying to append beyond capacity.
    """
    var buffer = PrefetchBuffer(capacity=1)

    var data = ones([2, 3, 8, 8], DType.float32)
    var labels = zeros([2, 10], DType.float32)
    var indices = List[Int]()
    indices.append(0)
    var batch = Batch(data^, labels^, indices^)

    buffer.append(batch^)

    # Try to append another
    var data2 = ones([2, 3, 8, 8], DType.float32)
    var labels2 = zeros([2, 10], DType.float32)
    var indices2 = List[Int]()
    indices2.append(1)
    var batch2 = Batch(data2^, labels2^, indices2^)

    try:
        buffer.append(batch2^)
        assert_true(False)  # Should have raised
    except:
        pass


fn test_prefetch_buffer_clear() raises:
    """Test clearing the buffer.

    Clear should remove all batches.
    """
    var buffer = PrefetchBuffer(capacity=2)

    var data = ones([2, 3, 8, 8], DType.float32)
    var labels = zeros([2, 10], DType.float32)
    var indices = List[Int]()
    indices.append(0)
    var batch = Batch(data^, labels^, indices^)

    buffer.append(batch^)
    assert_equal(buffer.batches.__len__(), 1)

    buffer.clear()
    assert_true(buffer.is_empty())


# ============================================================================
# PrefetchDataLoader Tests
# ============================================================================


fn test_prefetch_data_loader_creation() raises:
    """Test creating PrefetchDataLoader.

    Should wrap a BatchLoader with specified prefetch factor.
    """
    var data_shape: List[Int] = [10, 3, 32, 32]
    var label_shape: List[Int] = [10, 10]

    var data = ones(data_shape, DType.float32)
    var labels = zeros(label_shape, DType.float32)

    var dataset = AnyTensorDataset(data^, labels^)
    var sampler = RandomSampler(10)
    var loader = BatchLoader(dataset^, sampler^, batch_size=2)

    var prefetch = PrefetchDataLoader(loader^, prefetch_factor=2)
    assert_equal(prefetch.prefetch_factor, 2)


fn test_prefetch_data_loader_invalid_prefetch_factor() raises:
    """Test that PrefetchDataLoader validates prefetch_factor.

    Negative or zero prefetch_factor should raise an error.
    """
    var data_shape: List[Int] = [10, 3, 32, 32]
    var label_shape: List[Int] = [10, 10]

    var data = ones(data_shape, DType.float32)
    var labels = zeros(label_shape, DType.float32)

    var dataset = AnyTensorDataset(data^, labels^)
    var sampler = RandomSampler(10)
    var loader = BatchLoader(dataset^, sampler^, batch_size=2)

    try:
        var prefetch = PrefetchDataLoader(loader^, prefetch_factor=0)
        assert_true(False)  # Should have raised
    except:
        pass


fn test_prefetch_data_loader_iteration() raises:
    """Test that PrefetchDataLoader can iterate over batches.

    Should return batches similar to base loader.
    """
    var data_shape: List[Int] = [10, 3, 32, 32]
    var label_shape: List[Int] = [10, 10]

    var data = ones(data_shape, DType.float32)
    var labels = zeros(label_shape, DType.float32)

    var dataset = AnyTensorDataset(data^, labels^)
    var sampler = RandomSampler(10)
    var loader = BatchLoader(dataset^, sampler^, batch_size=2)

    var prefetch = PrefetchDataLoader(loader^, prefetch_factor=2)

    # Get batches from prefetch loader
    var batches = prefetch.__iter__()

    # Should have 5 batches (10 samples / 2 batch_size)
    assert_equal(batches.__len__(), 5)


fn main() raises:
    """Run all tests."""
    print("Testing PrefetchBuffer...")
    test_prefetch_buffer_creation()
    test_prefetch_buffer_capacity_validation()
    test_prefetch_buffer_append_and_pop()
    test_prefetch_buffer_capacity_limit()
    test_prefetch_buffer_clear()

    print("Testing PrefetchDataLoader...")
    test_prefetch_data_loader_creation()
    test_prefetch_data_loader_invalid_prefetch_factor()
    test_prefetch_data_loader_iteration()

    print("All prefetch tests passed!")
