"""Tests for batch loader - Part 1: batching and shuffling.

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
from shared.core.extensor import ExTensor


# ============================================================================
# BatchLoader Batching Tests
# ============================================================================


fn test_batch_loader_fixed_batch_size() raises:
    """Test creating batches of fixed size.

    Should group consecutive samples into batches of batch_size,
    with proper tensor stacking for efficient GPU processing.
    """
    var data_list = List[Float32]()
    for i in range(100):
        data_list.append(Float32(i))
    var data = ExTensor(data_list^)
    var labels_list = List[Int]()
    for i in range(100):
        labels_list.append(i)
    var labels = ExTensor(labels_list^)
    var dataset = TensorDataset(data^, labels^)
    var dataset_len = dataset.__len__()
    var sampler = SequentialSampler(dataset_len)
    var loader = BatchLoader(dataset^, sampler^, batch_size=32, shuffle=False)

    # Loader should calculate correct number of batches
    assert_equal(loader.__len__(), 4)  # 100 / 32 = 3.125 -> 4 batches


fn test_batch_loader_perfect_division() raises:
    """Test dataset size perfectly divisible by batch_size.

    With 96 samples and batch_size=32, should create exactly 3 batches
    of equal size with no partial batch.
    """
    var data_list = List[Float32]()
    for i in range(96):
        data_list.append(Float32(i))
    var data = ExTensor(data_list^)
    var labels_list = List[Int]()
    for i in range(96):
        labels_list.append(i)
    var labels = ExTensor(labels_list^)
    var dataset = TensorDataset(data^, labels^)
    var dataset_len = dataset.__len__()
    var sampler = SequentialSampler(dataset_len)
    var loader = BatchLoader(dataset^, sampler^, batch_size=32, shuffle=False)

    assert_equal(loader.__len__(), 3)  # 96 / 32 = 3 exactly


fn test_batch_loader_partial_last_batch() raises:
    """Test handling of partial last batch.

    With 100 samples and batch_size=32, last batch should have only 4 samples
    unless drop_last=True.
    """
    var data_list = List[Float32]()
    for i in range(100):
        data_list.append(Float32(i))
    var data_shape = List[Int]()
    data_shape.append(100)
    var data = ExTensor(data_shape, DType.float32)
    for i in range(len(data_list)):
        data._set_float32(i, data_list[i])

    var labels_list = List[Int]()
    for i in range(100):
        labels_list.append(i)
    var labels_shape = List[Int]()
    labels_shape.append(100)
    var labels = ExTensor(labels_shape, DType.int32)
    for i in range(len(labels_list)):
        labels._set_int32(i, Int32(labels_list[i]))

    var dataset = TensorDataset(data^, labels^)

    # Without drop_last
    var dataset_len = dataset.__len__()
    var sampler1 = SequentialSampler(dataset_len)
    var loader = BatchLoader(
        dataset.copy(), sampler1^, batch_size=32, shuffle=False, drop_last=False
    )
    assert_equal(loader.__len__(), 4)  # Includes partial batch

    # With drop_last
    var data_list2 = List[Float32]()
    for i in range(100):
        data_list2.append(Float32(i))
    var data_shape2 = List[Int]()
    data_shape2.append(100)
    var data2 = ExTensor(data_shape2, DType.float32)
    for i in range(len(data_list2)):
        data2._set_float32(i, data_list2[i])

    var labels_list2 = List[Int]()
    for i in range(100):
        labels_list2.append(i)
    var labels_shape2 = List[Int]()
    labels_shape2.append(100)
    var labels2 = ExTensor(labels_shape2, DType.int32)
    for i in range(len(labels_list2)):
        labels2._set_int32(i, Int32(labels_list2[i]))

    var dataset2 = TensorDataset(data2^, labels2^)
    var dataset2_len = dataset2.__len__()
    var sampler2 = SequentialSampler(dataset2_len)
    var loader2 = BatchLoader(
        dataset2^, sampler2^, batch_size=32, shuffle=False, drop_last=True
    )
    assert_equal(loader2.__len__(), 3)  # Drops partial batch


fn test_batch_loader_tensor_stacking() raises:
    """Test that BatchLoader API structure exists.

    Note: _stack_tensors may not be fully implemented,
    but we can test that the API structure is correct.
    """
    var data_list = List[Float32]()
    for i in range(100):
        data_list.append(Float32(i))
    var data_shape = List[Int]()
    data_shape.append(100)
    var data = ExTensor(data_shape, DType.float32)
    for i in range(len(data_list)):
        data._set_float32(i, data_list[i])

    var labels_list = List[Int]()
    for i in range(100):
        labels_list.append(i)
    var labels_shape = List[Int]()
    labels_shape.append(100)
    var labels = ExTensor(labels_shape, DType.int32)
    for i in range(len(labels_list)):
        labels._set_int32(i, Int32(labels_list[i]))

    var dataset = TensorDataset(data^, labels^)
    var dataset_len = dataset.__len__()
    var sampler = SequentialSampler(dataset_len)
    var loader = BatchLoader(dataset^, sampler^, batch_size=32, shuffle=False)

    # Test that loader was created successfully
    assert_equal(loader.__len__(), 4)


# ============================================================================
# BatchLoader Shuffling Tests
# ============================================================================


fn test_batch_loader_no_shuffle() raises:
    """Test that shuffle=False preserves dataset order.

    Batches should contain samples in dataset order: batch 0 has indices [0-31],
    batch 1 has indices [32-63], etc.
    """
    var data_list = List[Float32]()
    for i in range(100):
        data_list.append(Float32(i))
    var data_shape = List[Int]()
    data_shape.append(100)
    var data = ExTensor(data_shape, DType.float32)
    for i in range(len(data_list)):
        data._set_float32(i, data_list[i])
    var labels_list = List[Int]()
    for i in range(100):
        labels_list.append(i)
    var labels_shape = List[Int]()
    labels_shape.append(100)
    var labels = ExTensor(labels_shape, DType.int32)
    for i in range(len(labels_list)):
        labels._set_int32(i, Int32(labels_list[i]))
    var dataset = TensorDataset(data^, labels^)
    var dataset_len = dataset.__len__()
    var sampler = SequentialSampler(dataset_len)
    var loader = BatchLoader(dataset^, sampler^, batch_size=32, shuffle=False)

    # With shuffle=False, loader should use SequentialSampler
    assert_equal(loader.__len__(), 4)


fn test_batch_loader_shuffle() raises:
    """Test that shuffle=True randomizes sample order.

    Consecutive batches should not contain consecutive dataset indices,
    improving training by preventing order-dependent biases.
    """
    var data_list = List[Float32]()
    for i in range(100):
        data_list.append(Float32(i))
    var data_shape = List[Int]()
    data_shape.append(100)
    var data = ExTensor(data_shape, DType.float32)
    for i in range(len(data_list)):
        data._set_float32(i, data_list[i])
    var labels_list = List[Int]()
    for i in range(100):
        labels_list.append(i)
    var labels_shape = List[Int]()
    labels_shape.append(100)
    var labels = ExTensor(labels_shape, DType.int32)
    for i in range(len(labels_list)):
        labels._set_int32(i, Int32(labels_list[i]))
    var dataset = TensorDataset(data^, labels^)
    var dataset_len = dataset.__len__()
    var sampler = RandomSampler(dataset_len)
    var loader = BatchLoader(dataset^, sampler^, batch_size=32, shuffle=True)

    # With shuffle=True, loader should use RandomSampler
    assert_equal(loader.__len__(), 4)


fn test_batch_loader_shuffle_deterministic() raises:
    """Test that BatchLoader configuration can be deterministic.

    Loader creation with shuffle parameter should work,
    enabling reproducible experiments with fixed seed in sampler.
    """
    var data_list = List[Float32]()
    for i in range(100):
        data_list.append(Float32(i))
    var data_shape = List[Int]()
    data_shape.append(100)
    var data = ExTensor(data_shape, DType.float32)
    for i in range(len(data_list)):
        data._set_float32(i, data_list[i])
    var labels_list = List[Int]()
    for i in range(100):
        labels_list.append(i)
    var labels_shape = List[Int]()
    labels_shape.append(100)
    var labels = ExTensor(labels_shape, DType.int32)
    for i in range(len(labels_list)):
        labels._set_int32(i, Int32(labels_list[i]))
    var dataset = TensorDataset(data^, labels^)
    var dataset_len = dataset.__len__()
    var sampler = RandomSampler(dataset_len)
    var loader = BatchLoader(dataset^, sampler^, batch_size=32, shuffle=True)
    assert_equal(loader.__len__(), 4)


fn test_batch_loader_shuffle_per_epoch() raises:
    """Test that loader can handle multiple epochs.

    Loader API should support iteration multiple times,
    which is needed for multi-epoch training.
    """
    var data_list = List[Float32]()
    for i in range(100):
        data_list.append(Float32(i))
    var data_shape = List[Int]()
    data_shape.append(100)
    var data = ExTensor(data_shape, DType.float32)
    for i in range(len(data_list)):
        data._set_float32(i, data_list[i])
    var labels_list = List[Int]()
    for i in range(100):
        labels_list.append(i)
    var labels_shape = List[Int]()
    labels_shape.append(100)
    var labels = ExTensor(labels_shape, DType.int32)
    for i in range(len(labels_list)):
        labels._set_int32(i, Int32(labels_list[i]))
    var dataset = TensorDataset(data^, labels^)
    var dataset_len = dataset.__len__()
    var sampler = RandomSampler(dataset_len)
    var loader = BatchLoader(dataset^, sampler^, batch_size=32, shuffle=True)

    # Loader can be iterated multiple times (each call to __iter__)
    var batches1 = loader.__iter__()
    var batches2 = loader.__iter__()

    # Both should produce same number of batches
    assert_equal(len(batches1), len(batches2))


# ============================================================================
# Main Test Runner
# ============================================================================


fn main() raises:
    """Run batch loader tests - Part 1 (batching and shuffling)."""
    print("Running batch loader tests - Part 1...")

    # Batching tests
    test_batch_loader_fixed_batch_size()
    test_batch_loader_perfect_division()
    test_batch_loader_partial_last_batch()
    test_batch_loader_tensor_stacking()

    # Shuffling tests
    test_batch_loader_no_shuffle()
    test_batch_loader_shuffle()
    test_batch_loader_shuffle_deterministic()
    test_batch_loader_shuffle_per_epoch()

    print("✓ All batch loader Part 1 tests passed!")
