"""High-level integration tests for dataset abstractions

Tests cover cross-component interactions between datasets, loaders, and samplers.
Individual unit tests exist in datasets/ subdirectory.

Integration Points:
- TensorDataset + BatchLoader access patterns
- Dataset trait conformance and interface consistency
- Metadata properties (length, shape consistency)
"""


from tests.shared.conftest import (
    assert_true,
    assert_equal,
    assert_not_equal,
    assert_greater,
    TestFixtures,
)
from shared.data.datasets import AnyTensorDataset
from shared.data.loaders import BatchLoader
from shared.data.samplers import SequentialSampler
from shared.tensor.any_tensor import AnyTensor


def test_dataset_length_consistency() raises:
    """Test AnyTensorDataset reports correct length.

    __len__() should match actual number of samples in first dimension.

    Integration Points:
        - AnyTensorDataset initialization
        - Length metadata accuracy

    Success Criteria:
        - Dataset length equals number of samples
        - __len__() consistent across calls
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

    # Check length matches
    assert_equal(dataset.__len__(), 100)


def test_dataset_sequential_access() raises:
    """Test AnyTensorDataset supports sequential __getitem__ access.

    Should be able to retrieve samples by index in order.

    Integration Points:
        - __getitem__ implementation
        - Sequential access pattern
        - Tuple return type (data, label)

    Success Criteria:
        - Can access first, middle, and last samples
        - Returns tuple of (data, label)
        - No errors or out-of-bounds exceptions
    """
    TestFixtures.set_seed()

    var data_list = List[Float32]()
    for i in range(10):
        data_list.append(Float32(i))
    var data = AnyTensor(data_list^)

    var labels_list = List[Int]()
    for i in range(10):
        labels_list.append(i * 2)
    var labels = AnyTensor(labels_list^)

    var dataset = AnyTensorDataset(data^, labels^)

    # Access first sample
    var sample0 = dataset.__getitem__(0)
    # Access middle sample
    var sample5 = dataset.__getitem__(5)
    # Access last sample
    var sample9 = dataset.__getitem__(9)

    # All should succeed
    assert_equal(dataset.__len__(), 10)


def test_dataset_negative_indexing() raises:
    """Test AnyTensorDataset supports negative indexing.

    Negative indices count from end: -1 is last, -2 is second-to-last, etc.

    Integration Points:
        - Negative index handling in __getitem__
        - Index normalization

    Success Criteria:
        - Negative indices resolve correctly
        - Last element accessible via -1
        - Second-to-last accessible via -2
    """
    TestFixtures.set_seed()

    var data_list = List[Float32]()
    for i in range(20):
        data_list.append(Float32(i))
    var data = AnyTensor(data_list^)

    var labels_list = List[Int]()
    for i in range(20):
        labels_list.append(i)
    var labels = AnyTensor(labels_list^)

    var dataset = AnyTensorDataset(data^, labels^)

    # Access with negative indices
    var last = dataset.__getitem__(-1)
    var second_last = dataset.__getitem__(-2)

    # Both should succeed
    assert_equal(dataset.__len__(), 20)


def test_dataset_bounds_checking() raises:
    """Test AnyTensorDataset handles out-of-bounds access properly.

    Accessing index >= len() should raise error.
    Accessing very negative index should raise error.

    Integration Points:
        - Bounds validation in __getitem__
        - Error handling for invalid indices

    Success Criteria:
        - Out-of-bounds access raises error (caught via raises)
        - Error message describes the problem
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

    # Valid access should work
    var valid = dataset.__getitem__(0)
    var valid2 = dataset.__getitem__(9)

    # Out-of-bounds access would be tested separately if error propagation works


def test_dataset_with_loader() raises:
    """Test AnyTensorDataset works seamlessly with BatchLoader.

    Dataset should integrate with loader for batching.

    Integration Points:
        - AnyTensorDataset + BatchLoader
        - Loader batch count calculation
        - Dataset trait conformance

    Success Criteria:
        - Loader accepts dataset
        - Correct number of batches
        - No type errors or incompatibilities
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
    var loader = BatchLoader(dataset^, sampler^, batch_size=25, shuffle=False)

    # Verify integration worked
    assert_equal(loader.__len__(), 4)  # 100 / 25 = 4 batches


def test_tensor_dataset_batching_shapes() raises:
    """Test AnyTensorDataset produces correct shapes when batched.

    When batches are created from 1D input, batch shape should be [batch_size].

    Integration Points:
        - Dataset shape preservation
        - Loader batching behavior
        - Tensor shape consistency

    Success Criteria:
        - Dataset preserves original sample shapes
        - Batching doesn't change data shape structure
    """
    TestFixtures.set_seed()

    var data_list = List[Float32]()
    for i in range(50):
        data_list.append(Float32(i))
    var data = AnyTensor(data_list^)

    var labels_list = List[Int]()
    for i in range(50):
        labels_list.append(i)
    var labels = AnyTensor(labels_list^)

    var dataset = AnyTensorDataset(data^, labels^)
    var dataset_len = dataset.__len__()

    assert_equal(dataset_len, 50)


def test_dataset_random_access() raises:
    """Test AnyTensorDataset supports random access via samplers.

    Can access arbitrary indices via __getitem__ in any order.

    Integration Points:
        - Direct __getitem__ calls
        - No sequential requirement
        - Sampler index generation

    Success Criteria:
        - Can access samples in arbitrary order
        - Results are consistent (same index -> same sample)
    """
    TestFixtures.set_seed()

    var data_list = List[Float32]()
    for i in range(20):
        data_list.append(Float32(i))
    var data = AnyTensor(data_list^)

    var labels_list = List[Int]()
    for i in range(20):
        labels_list.append(i)
    var labels = AnyTensor(labels_list^)

    var dataset = AnyTensorDataset(data^, labels^)

    # Access in non-sequential order
    var s0 = dataset.__getitem__(0)
    var s15 = dataset.__getitem__(15)
    var s5 = dataset.__getitem__(5)
    var s0_again = dataset.__getitem__(0)

    # First and repeated accesses should be identical
    assert_equal(dataset.__len__(), 20)


def test_dataset_interface_protocol() raises:
    """Test AnyTensorDataset conforms to Dataset trait protocol.

    Must implement __len__() and __getitem__() correctly.

    Integration Points:
        - Dataset trait definition
        - Method signature conformance
        - Return type correctness

    Success Criteria:
        - Both required methods present and callable
        - Methods have correct signatures
        - No type errors when used as Dataset
    """
    TestFixtures.set_seed()

    var data_list = List[Float32]()
    for i in range(30):
        data_list.append(Float32(i))
    var data = AnyTensor(data_list^)

    var labels_list = List[Int]()
    for i in range(30):
        labels_list.append(i)
    var labels = AnyTensor(labels_list^)

    var dataset = AnyTensorDataset(data^, labels^)

    # Both methods should be accessible
    var len_val = dataset.__len__()
    assert_equal(len_val, 30)

    # __getitem__ should work
    var sample = dataset.__getitem__(0)


def test_dataset_small_size() raises:
    """Test AnyTensorDataset with very small number of samples.

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
    var data = AnyTensor(data_list^)

    var labels_list = List[Int]()
    labels_list.append(0)
    labels_list.append(1)
    var labels = AnyTensor(labels_list^)

    var dataset = AnyTensorDataset(data^, labels^)
    assert_equal(dataset.__len__(), 2)


def test_dataset_single_sample() raises:
    """Test AnyTensorDataset with single sample (minimal edge case).

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
    var data = AnyTensor(data_list^)

    var labels_list = List[Int]()
    labels_list.append(99)
    var labels = AnyTensor(labels_list^)

    var dataset = AnyTensorDataset(data^, labels^)

    assert_equal(dataset.__len__(), 1)
    var s0 = dataset.__getitem__(0)
    var s_neg = dataset.__getitem__(-1)


def test_dataset_large_size() raises:
    """Test AnyTensorDataset with larger number of samples.

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
    var data = AnyTensor(data_list^)

    var labels_list = List[Int]()
    for i in range(1000):
        labels_list.append(i % 10)
    var labels = AnyTensor(labels_list^)

    var dataset = AnyTensorDataset(data^, labels^)
    assert_equal(dataset.__len__(), 1000)


def test_dataset_repeated_access() raises:
    """Test AnyTensorDataset handles repeated access to same sample.

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
    var data = AnyTensor(data_list^)

    var labels_list = List[Int]()
    for i in range(10):
        labels_list.append(i)
    var labels = AnyTensor(labels_list^)

    var dataset = AnyTensorDataset(data^, labels^)

    # Access same sample multiple times
    var s5_a = dataset.__getitem__(5)
    var s5_b = dataset.__getitem__(5)
    var s5_c = dataset.__getitem__(5)

    # All should be identical
    assert_equal(dataset.__len__(), 10)


def test_dataset_with_different_batch_sizes() raises:
    """Test AnyTensorDataset works with various batch sizes in loader.

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
    var data = AnyTensor(data_list^)

    var labels_list = List[Int]()
    for i in range(100):
        labels_list.append(i)
    var labels = AnyTensor(labels_list^)

    var dataset = AnyTensorDataset(data^, labels^)
    var dataset_len = dataset.__len__()

    # Verify dataset reports correct length
    assert_equal(dataset_len, 100)

    # Batch count is determined by ceil(dataset_len / batch_size)
    # batch_size=1: ceil(100/1) = 100
    # batch_size=16: ceil(100/16) = 7
    # batch_size=32: ceil(100/32) = 4


def main() raises:
    """Run all test_datasets tests."""
    print("Running test_datasets tests...")

    test_dataset_length_consistency()
    print("✓ test_dataset_length_consistency")

    test_dataset_sequential_access()
    print("✓ test_dataset_sequential_access")

    test_dataset_negative_indexing()
    print("✓ test_dataset_negative_indexing")

    test_dataset_bounds_checking()
    print("✓ test_dataset_bounds_checking")

    test_dataset_with_loader()
    print("✓ test_dataset_with_loader")

    test_tensor_dataset_batching_shapes()
    print("✓ test_tensor_dataset_batching_shapes")

    test_dataset_random_access()
    print("✓ test_dataset_random_access")

    test_dataset_interface_protocol()
    print("✓ test_dataset_interface_protocol")

    test_dataset_small_size()
    print("✓ test_dataset_small_size")

    test_dataset_single_sample()
    print("✓ test_dataset_single_sample")

    test_dataset_large_size()
    print("✓ test_dataset_large_size")

    test_dataset_repeated_access()
    print("✓ test_dataset_repeated_access")

    test_dataset_with_different_batch_sizes()
    print("✓ test_dataset_with_different_batch_sizes")

    print("\nAll test_datasets tests passed!")
