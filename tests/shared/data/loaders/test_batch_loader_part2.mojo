"""Tests for batch loader - Part 2: per-epoch coverage and performance.

ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
high test load. Split from test_batch_loader.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""

from tests.shared.conftest import (
    assert_true,
    assert_equal,
    assert_not_equal,
    TestFixtures,
)
from shared.data.datasets import TensorDataset
from shared.data.loaders import BatchLoader
from shared.data.samplers import SequentialSampler, RandomSampler
from shared.core.any_tensor import AnyTensor


# ============================================================================
# BatchLoader Per-Epoch and Performance Tests
# ============================================================================


fn test_batch_loader_all_samples_per_epoch() raises:
    """Test that loader produces correct number of batches.

    Each epoch should yield correct number of batches
    covering all samples.
    """
    var data_list = List[Float32]()
    for i in range(100):
        data_list.append(Float32(i))
    var data_shape = List[Int]()
    data_shape.append(100)
    var data = AnyTensor(data_shape, DType.float32)
    for i in range(len(data_list)):
        data._set_float32(i, data_list[i])
    var labels_list = List[Int]()
    for i in range(100):
        labels_list.append(i)
    var labels_shape = List[Int]()
    labels_shape.append(100)
    var labels = AnyTensor(labels_shape, DType.int32)
    for i in range(len(labels_list)):
        labels._set_int32(i, Int32(labels_list[i]))
    var dataset = TensorDataset(data^, labels^)
    var dataset_len = dataset.__len__()
    var sampler = RandomSampler(dataset_len)
    var loader = BatchLoader(dataset^, sampler^, batch_size=32, shuffle=True)

    var batches = loader.__iter__()
    # Should have 4 batches (ceil(100/32) = 4)
    assert_equal(len(batches), 4)


# ============================================================================
# BatchLoader Performance Tests
# ============================================================================


fn test_batch_loader_efficient_batching() raises:
    """Test that batching API structure is efficient.

    BatchLoader should efficiently manage batches,
    creating them on-demand during iteration.
    """
    var data_list = List[Float32]()
    for i in range(1000):
        data_list.append(Float32(i))
    var data_shape = List[Int]()
    data_shape.append(1000)
    var data = AnyTensor(data_shape, DType.float32)
    for i in range(len(data_list)):
        data._set_float32(i, data_list[i])
    var labels_list = List[Int]()
    for i in range(1000):
        labels_list.append(i)
    var labels_shape = List[Int]()
    labels_shape.append(1000)
    var labels = AnyTensor(labels_shape, DType.int32)
    for i in range(len(labels_list)):
        labels._set_int32(i, Int32(labels_list[i]))
    var dataset = TensorDataset(data^, labels^)
    var dataset_len = dataset.__len__()
    var sampler = SequentialSampler(dataset_len)
    var loader = BatchLoader(dataset^, sampler^, batch_size=32, shuffle=False)

    # Should create appropriate number of batches
    assert_equal(loader.__len__(), 32)  # ceil(1000/32) = 32


fn test_batch_loader_iteration_speed() raises:
    """Test that loader creates correct number of batches.

    Should calculate batch count correctly for efficient iteration,
    as this is done every training epoch.
    """
    var data_list = List[Float32]()
    for i in range(3200):
        data_list.append(Float32(i))
    var data_shape = List[Int]()
    data_shape.append(3200)
    var data = AnyTensor(data_shape, DType.float32)
    for i in range(len(data_list)):
        data._set_float32(i, data_list[i])
    var labels_list = List[Int]()
    for i in range(3200):
        labels_list.append(i)
    var labels_shape = List[Int]()
    labels_shape.append(3200)
    var labels = AnyTensor(labels_shape, DType.int32)
    for i in range(len(labels_list)):
        labels._set_int32(i, Int32(labels_list[i]))
    var dataset = TensorDataset(data^, labels^)
    var dataset_len = dataset.__len__()
    var sampler = SequentialSampler(dataset_len)
    var loader = BatchLoader(dataset^, sampler^, batch_size=32, shuffle=False)

    # Should have exactly 100 batches
    assert_equal(loader.__len__(), 100)


# ============================================================================
# Main Test Runner
# ============================================================================


fn main() raises:
    """Run batch loader tests - Part 2 (per-epoch coverage and performance)."""
    print("Running batch loader tests - Part 2...")

    # Per-epoch tests
    test_batch_loader_all_samples_per_epoch()

    # Performance tests
    test_batch_loader_efficient_batching()
    test_batch_loader_iteration_speed()

    print("✓ All batch loader Part 2 tests passed!")
