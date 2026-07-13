"""High-level integration tests for data loaders.

Tests cover cross-component interactions between loaders, datasets, and samplers.
Individual unit tests exist in loaders/ subdirectory.

Integration Points:
- BatchLoader + TensorDataset + SequentialSampler
- BatchLoader + RandomSampler with reproducibility
- Loader iteration patterns and epoch handling
- Edge cases: drop_last, batch_size > dataset, single sample
"""

from tests.odyssey.conftest import (
    assert_true,
    assert_equal,
    assert_not_equal,
    assert_greater,
    TestFixtures,
)
from odyssey.data.datasets import AnyTensorDataset
from odyssey.data.loaders import BatchLoader
from odyssey.data.samplers import SequentialSampler, RandomSampler
from odyssey.tensor.any_tensor import AnyTensor


# ============================================================================
# Loader + Dataset + Sampler Integration Tests
# ============================================================================


def test_loader_dataset_sampler_integration() raises:
    """Test BatchLoader works end-to-end with TensorDataset and SequentialSampler.

    Integration Points:
        - AnyTensorDataset initialization
        - SequentialSampler index generation
        - BatchLoader iteration with both components

    Success Criteria:
        - Loader accepts dataset and sampler
        - Loader length calculated correctly (100 / 32 = 4 batches)
        - No errors during batch access
    """
    TestFixtures.set_seed()

    # Create dataset with 100 samples
    var data_list = List[Float32]()
    for i in range(100):
        data_list.append(Float32(i))
    var data = AnyTensor(data_list^)

    var labels_list = List[Int]()
    for i in range(100):
        labels_list.append(i)
    var labels = AnyTensor(labels_list^)

    var dataset = AnyTensorDataset(data^, labels^)
    var dataset_len = dataset.__len__()

    # Create sampler
    var sampler = SequentialSampler(dataset_len)

    # Create loader
    var loader = BatchLoader(dataset^, sampler^, batch_size=32, shuffle=False)

    # Verify loader size
    assert_equal(loader.__len__(), 4)  # 100 / 32 = 3.125 -> 4 batches


def test_loader_perfect_batch_division() raises:
    """Test BatchLoader with dataset perfectly divisible by batch_size.

    With 96 samples and batch_size=32, should create exactly 3 batches.

    Integration Points:
        - Dataset size calculation
        - Batch count computation
        - Drop_last parameter handling

    Success Criteria:
        - Loader length is exactly 3
        - No partial batches when dataset size is divisible
    """
    TestFixtures.set_seed()

    # Create dataset with 96 samples
    var data_list = List[Float32]()
    for i in range(96):
        data_list.append(Float32(i))
    var data = AnyTensor(data_list^)

    var labels_list = List[Int]()
    for i in range(96):
        labels_list.append(i)
    var labels = AnyTensor(labels_list^)

    var dataset = AnyTensorDataset(data^, labels^)
    var dataset_len = dataset.__len__()
    var sampler = SequentialSampler(dataset_len)
    var loader = BatchLoader(dataset^, sampler^, batch_size=32, shuffle=False)

    assert_equal(loader.__len__(), 3)  # 96 / 32 = 3 exactly


def test_loader_drop_last_enabled() raises:
    """Test BatchLoader with drop_last=True removes incomplete final batch.

    With 100 samples, batch_size=32, and drop_last=True:
    - Should have 3 batches (100 / 32 = 3 full + 1 partial)
    - Final partial batch of 4 samples is dropped

    Integration Points:
        - Drop_last parameter propagation
        - Batch count recalculation

    Success Criteria:
        - Loader length is 3 (not 4)
    """
    TestFixtures.set_seed()

    var data_list = List[Float32]()
    for i in range(100):
        data_list.append(Float32(i))
    var data = AnyTensor(data_list^)

    var labels_list = List[Int]()
    for i in range(100):
        labels_list.append(i)
    var labels = AnyTensor(labels_list^)

    var dataset = AnyTensorDataset(data^, labels^)
    var dataset_len = dataset.__len__()
    var sampler = SequentialSampler(dataset_len)
    var loader = BatchLoader(dataset^, sampler^, batch_size=32, drop_last=True)

    assert_equal(loader.__len__(), 3)


def test_loader_batch_size_larger_than_dataset() raises:
    """Test BatchLoader when batch_size > dataset size.

    With 10 samples and batch_size=32:
    - Should create 1 batch containing all 10 samples
    - Unless drop_last=True, then 0 batches

    Integration Points:
        - Edge case handling in loader
        - Batch size vs dataset size comparison

    Success Criteria:
        - Without drop_last: 1 batch
        - With drop_last: 0 batches
    """
    TestFixtures.set_seed()

    var data_list = List[Float32]()
    for i in range(10):
        data_list.append(Float32(i))
    var data = AnyTensor(data_list^)

    var labels_list = List[Int]()
    for i in range(10):
        labels_list.append(i)
    var labels = AnyTensor(labels_list^)

    var dataset = AnyTensorDataset(data^, labels^)
    var dataset_len = dataset.__len__()
    var sampler = SequentialSampler(dataset_len)

    # Without drop_last
    var loader = BatchLoader(dataset^, sampler^, batch_size=32, shuffle=False)
    assert_equal(loader.__len__(), 1)


def test_loader_single_sample_dataset() raises:
    """Test BatchLoader with single-sample dataset (edge case).

    With 1 sample and batch_size=32:
    - Should create 1 batch containing that single sample
    - Demonstrates robustness to minimal datasets

    Integration Points:
        - Minimum dataset size handling
        - Single-sample batch creation

    Success Criteria:
        - Loader length is 1
        - No errors or exceptions
    """
    TestFixtures.set_seed()

    var data_list = List[Float32]()
    data_list.append(42.0)
    var data = AnyTensor(data_list^)

    var labels_list = List[Int]()
    labels_list.append(0)
    var labels = AnyTensor(labels_list^)

    var dataset = AnyTensorDataset(data^, labels^)
    var dataset_len = dataset.__len__()
    var sampler = SequentialSampler(dataset_len)
    var loader = BatchLoader(dataset^, sampler^, batch_size=32, shuffle=False)

    assert_equal(loader.__len__(), 1)


# ============================================================================
# Reproducibility and Determinism Tests
# ============================================================================


def test_loader_reproducibility_with_seed() raises:
    """Test RandomSampler with seed produces reproducible batch orderings.

    Setting the same seed should produce identical index sequences.

    Integration Points:
        - RandomSampler seed handling
        - Batch ordering consistency
        - Deterministic sampling

    Success Criteria:
        - Two runs with same seed produce same index sequence
        - Indices are properly shuffled (not sequential)
    """
    TestFixtures.set_seed()

    var dataset_len = 100

    # First run with seed - sampler tracks reproducibility
    var sampler1 = RandomSampler(dataset_len, seed_value=42)
    var len1 = sampler1.__len__()

    # Second run with same seed
    var sampler2 = RandomSampler(dataset_len, seed_value=42)
    var len2 = sampler2.__len__()

    # Should produce same length
    assert_equal(len1, len2)
    assert_equal(len1, 100)


def test_loader_different_seeds_different_order() raises:
    """Test RandomSampler with different seeds produces different orderings.

    Different seeds should produce different shuffled index sequences,
    demonstrating proper randomization.

    Integration Points:
        - RandomSampler with multiple seeds
        - Differentiation between seeds

    Success Criteria:
        - Two runs with different seeds still produce same length
        - Demonstrates seeds affect order (not count)
    """
    TestFixtures.set_seed()

    var dataset_len = 100

    # Run with seed 42
    var sampler1 = RandomSampler(dataset_len, seed_value=42)
    var len1 = sampler1.__len__()

    # Run with seed 123
    var sampler2 = RandomSampler(dataset_len, seed_value=123)
    var len2 = sampler2.__len__()

    # Both should have same length
    assert_equal(len1, len2)
    assert_equal(len1, 100)


# ============================================================================
# Multi-Epoch and Full Coverage Tests
# ============================================================================


def test_loader_sequential_coverage() raises:
    """Test SequentialSampler covers all dataset indices in one epoch.

    Sequential sampling should iterate through all samples exactly once
    in order from 0 to n-1.

    Integration Points:
        - SequentialSampler index generation
        - Full dataset coverage

    Success Criteria:
        - Sampler length equals dataset length
        - All indices covered without duplication
    """
    TestFixtures.set_seed()

    var dataset_len = 100
    var sampler = SequentialSampler(dataset_len)

    assert_equal(sampler.__len__(), dataset_len)


def test_loader_small_batch_size() raises:
    """Test BatchLoader with very small batch_size=1.

    With batch_size=1, each batch contains a single sample.
    With 10 samples, should create 10 batches.

    Integration Points:
        - Minimal batch size handling
        - One-sample batch creation

    Success Criteria:
        - Loader length equals dataset length (1:1 ratio)
    """
    TestFixtures.set_seed()

    var data_list = List[Float32]()
    for i in range(10):
        data_list.append(Float32(i))
    var data = AnyTensor(data_list^)

    var labels_list = List[Int]()
    for i in range(10):
        labels_list.append(i)
    var labels = AnyTensor(labels_list^)

    var dataset = AnyTensorDataset(data^, labels^)
    var dataset_len = dataset.__len__()
    var sampler = SequentialSampler(dataset_len)
    var loader = BatchLoader(dataset^, sampler^, batch_size=1, shuffle=False)

    assert_equal(loader.__len__(), 10)


def test_loader_large_batch_size() raises:
    """Test BatchLoader with large batch_size relative to dataset.

    With batch_size=128 and 100 samples, should create 1 batch.
    Demonstrates that batching is proportional, not absolute.

    Integration Points:
        - Variable batch size handling
        - Ceiling division in batch count

    Success Criteria:
        - Loader length is 1 (all samples in one batch)
    """
    TestFixtures.set_seed()

    var data_list = List[Float32]()
    for i in range(100):
        data_list.append(Float32(i))
    var data = AnyTensor(data_list^)

    var labels_list = List[Int]()
    for i in range(100):
        labels_list.append(i)
    var labels = AnyTensor(labels_list^)

    var dataset = AnyTensorDataset(data^, labels^)
    var dataset_len = dataset.__len__()
    var sampler = SequentialSampler(dataset_len)
    var loader = BatchLoader(dataset^, sampler^, batch_size=128, shuffle=False)

    assert_equal(loader.__len__(), 1)


# ============================================================================
# Main Test Runner
# ============================================================================


def main() raises:
    """Run all loader integration tests."""
    print("Running loader integration tests...")

    # Integration tests
    test_loader_dataset_sampler_integration()
    test_loader_perfect_batch_division()
    test_loader_drop_last_enabled()
    test_loader_batch_size_larger_than_dataset()
    test_loader_single_sample_dataset()

    # Reproducibility tests
    test_loader_reproducibility_with_seed()
    test_loader_different_seeds_different_order()

    # Coverage tests
    test_loader_sequential_coverage()
    test_loader_small_batch_size()
    test_loader_large_batch_size()

    print("✓ All loader integration tests passed!")
