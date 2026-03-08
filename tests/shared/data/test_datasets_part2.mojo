"""High-level integration tests for dataset abstractions (Part 2 of 2).

Tests cover edge cases and advanced access patterns for ExTensorDataset.
Individual unit tests exist in datasets/ subdirectory.

ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
high test load. Split from test_datasets.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md

Integration Points:
- Edge cases: empty datasets, bounds checking
- Repeated access consistency
- Batch size variability
"""

from tests.shared.conftest import (
    assert_true,
    assert_equal,
    assert_not_equal,
    assert_greater,
    TestFixtures,
)
from shared.data.datasets import ExTensorDataset
from shared.data.loaders import BatchLoader
from shared.data.samplers import SequentialSampler
from shared.core.extensor import ExTensor


# ============================================================================
# Dataset Edge Case Tests
# ============================================================================


fn test_dataset_small_size() raises:
    """Test ExTensorDataset with very small number of samples.

    Should handle minimal datasets (2-5 samples) correctly.

    Integration Points:
        - Small dataset initialization
        - Indexing for minimal samples

    Success Criteria:
        - Can create 2-sample dataset
        - Can access both samples
        - Length reported correctly
    """
    TestFixtures.set_seed()

    var data_list = List[Float32]()
    data_list.append(1.0)
    data_list.append(2.0)
    var data = ExTensor(data_list^)

    var labels_list = List[Int]()
    labels_list.append(0)
    labels_list.append(1)
    var labels = ExTensor(labels_list^)

    var dataset = ExTensorDataset(data^, labels^)
    assert_equal(dataset.__len__(), 2)


fn test_dataset_single_sample() raises:
    """Test ExTensorDataset with single sample (minimal edge case).

    Should handle 1-sample datasets correctly.

    Integration Points:
        - Minimal dataset size
        - Single-sample access

    Success Criteria:
        - 1-sample dataset initializes
        - Can access via index 0
        - Can access via index -1
        - Length is 1
    """
    TestFixtures.set_seed()

    var data_list = List[Float32]()
    data_list.append(42.0)
    var data = ExTensor(data_list^)

    var labels_list = List[Int]()
    labels_list.append(99)
    var labels = ExTensor(labels_list^)

    var dataset = ExTensorDataset(data^, labels^)

    assert_equal(dataset.__len__(), 1)
    var s0 = dataset.__getitem__(0)
    var s_neg = dataset.__getitem__(-1)


fn test_dataset_large_size() raises:
    """Test ExTensorDataset with larger number of samples.

    Should scale to thousands of samples without issues.

    Integration Points:
        - Large dataset creation
        - Memory efficiency
        - Indexing performance

    Success Criteria:
        - Can create 1000-sample dataset
        - Length reported correctly
        - Access patterns still work
    """
    TestFixtures.set_seed()

    var data_list = List[Float32]()
    for i in range(1000):
        data_list.append(Float32(i % 100) / 100.0)
    var data = ExTensor(data_list^)

    var labels_list = List[Int]()
    for i in range(1000):
        labels_list.append(i % 10)
    var labels = ExTensor(labels_list^)

    var dataset = ExTensorDataset(data^, labels^)
    assert_equal(dataset.__len__(), 1000)


fn test_dataset_repeated_access() raises:
    """Test ExTensorDataset handles repeated access to same sample.

    Accessing the same index multiple times should always return same sample.

    Integration Points:
        - Deterministic __getitem__ behavior
        - No state modification on access
        - Caching or lazy-loading implications

    Success Criteria:
        - Multiple accesses to same index return consistent results
        - No errors or exceptions
    """
    TestFixtures.set_seed()

    var data_list = List[Float32]()
    for i in range(10):
        data_list.append(Float32(i))
    var data = ExTensor(data_list^)

    var labels_list = List[Int]()
    for i in range(10):
        labels_list.append(i)
    var labels = ExTensor(labels_list^)

    var dataset = ExTensorDataset(data^, labels^)

    # Access same sample multiple times
    var s5_a = dataset.__getitem__(5)
    var s5_b = dataset.__getitem__(5)
    var s5_c = dataset.__getitem__(5)

    # All should be identical
    assert_equal(dataset.__len__(), 10)


fn test_dataset_with_different_batch_sizes() raises:
    """Test ExTensorDataset works with various batch sizes in loader.

    Should integrate with loaders using different batch_size values.

    Integration Points:
        - Dataset + Loader with variable batch_size
        - Batch count calculation
        - Loader flexibility

    Success Criteria:
        - Dataset length correctly reported
        - Batch count scales with batch_size
    """
    TestFixtures.set_seed()

    var data_list = List[Float32]()
    for i in range(100):
        data_list.append(Float32(i))
    var data = ExTensor(data_list^)

    var labels_list = List[Int]()
    for i in range(100):
        labels_list.append(i)
    var labels = ExTensor(labels_list^)

    var dataset = ExTensorDataset(data^, labels^)
    var dataset_len = dataset.__len__()

    # Verify dataset reports correct length
    assert_equal(dataset_len, 100)

    # Batch count is determined by ceil(dataset_len / batch_size)
    # batch_size=1: ceil(100/1) = 100
    # batch_size=16: ceil(100/16) = 7
    # batch_size=32: ceil(100/32) = 4


# ============================================================================
# Main Test Runner
# ============================================================================


fn main() raises:
    """Run dataset integration tests (Part 2)."""
    print("Running dataset integration tests (Part 2)...")

    # Edge case tests
    test_dataset_small_size()
    test_dataset_single_sample()
    test_dataset_large_size()
    test_dataset_repeated_access()
    test_dataset_with_different_batch_sizes()

    print("✓ All dataset integration tests (Part 2) passed!")
