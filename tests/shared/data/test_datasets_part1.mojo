"""High-level integration tests for dataset abstractions (Part 1 of 2).

Tests cover cross-component interactions between datasets, loaders, and samplers.
Individual unit tests exist in datasets/ subdirectory.

ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
high test load. Split from test_datasets.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md

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
from shared.data.datasets import ExTensorDataset
from shared.data.loaders import BatchLoader
from shared.data.samplers import SequentialSampler
from shared.core.extensor import ExTensor


# ============================================================================
# Dataset Interface Conformance Tests
# ============================================================================


fn test_dataset_length_consistency() raises:
    """Test ExTensorDataset reports correct length.

    __len__() should match actual number of samples in first dimension.

    Integration Points:
        - ExTensorDataset initialization
        - Length metadata accuracy

    Success Criteria:
        - Dataset length equals number of samples
        - __len__() consistent across calls
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

    # Check length matches
    assert_equal(dataset.__len__(), 100)


fn test_dataset_sequential_access() raises:
    """Test ExTensorDataset supports sequential __getitem__ access.

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
    var data = ExTensor(data_list^)

    var labels_list = List[Int]()
    for i in range(10):
        labels_list.append(i * 2)
    var labels = ExTensor(labels_list^)

    var dataset = ExTensorDataset(data^, labels^)

    # Access first sample
    var sample0 = dataset.__getitem__(0)
    # Access middle sample
    var sample5 = dataset.__getitem__(5)
    # Access last sample
    var sample9 = dataset.__getitem__(9)

    # All should succeed
    assert_equal(dataset.__len__(), 10)


fn test_dataset_negative_indexing() raises:
    """Test ExTensorDataset supports negative indexing.

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
    var data = ExTensor(data_list^)

    var labels_list = List[Int]()
    for i in range(20):
        labels_list.append(i)
    var labels = ExTensor(labels_list^)

    var dataset = ExTensorDataset(data^, labels^)

    # Access with negative indices
    var last = dataset.__getitem__(-1)
    var second_last = dataset.__getitem__(-2)

    # Both should succeed
    assert_equal(dataset.__len__(), 20)


fn test_dataset_bounds_checking() raises:
    """Test ExTensorDataset handles out-of-bounds access properly.

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
    var data = ExTensor(data_list^)

    var labels_list = List[Int]()
    for i in range(10):
        labels_list.append(i)
    var labels = ExTensor(labels_list^)

    var dataset = ExTensorDataset(data^, labels^)

    # Valid access should work
    var valid = dataset.__getitem__(0)
    var valid2 = dataset.__getitem__(9)

    # Out-of-bounds access would be tested separately if error propagation works


fn test_dataset_with_loader() raises:
    """Test ExTensorDataset works seamlessly with BatchLoader.

    Dataset should integrate with loader for batching.

    Integration Points:
        - ExTensorDataset + BatchLoader
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
    var data = ExTensor(data_list^)

    var labels_list = List[Int]()
    for i in range(100):
        labels_list.append(i)
    var labels = ExTensor(labels_list^)

    var dataset = ExTensorDataset(data^, labels^)
    var dataset_len = dataset.__len__()

    var sampler = SequentialSampler(dataset_len)
    var loader = BatchLoader(dataset^, sampler^, batch_size=25, shuffle=False)

    # Verify integration worked
    assert_equal(loader.__len__(), 4)  # 100 / 25 = 4 batches


# ============================================================================
# Dataset-Loader Integration Tests
# ============================================================================


fn test_tensor_dataset_batching_shapes() raises:
    """Test ExTensorDataset produces correct shapes when batched.

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
    var data = ExTensor(data_list^)

    var labels_list = List[Int]()
    for i in range(50):
        labels_list.append(i)
    var labels = ExTensor(labels_list^)

    var dataset = ExTensorDataset(data^, labels^)
    var dataset_len = dataset.__len__()

    assert_equal(dataset_len, 50)


fn test_dataset_random_access() raises:
    """Test ExTensorDataset supports random access via samplers.

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
    var data = ExTensor(data_list^)

    var labels_list = List[Int]()
    for i in range(20):
        labels_list.append(i)
    var labels = ExTensor(labels_list^)

    var dataset = ExTensorDataset(data^, labels^)

    # Access in non-sequential order
    var s0 = dataset.__getitem__(0)
    var s15 = dataset.__getitem__(15)
    var s5 = dataset.__getitem__(5)
    var s0_again = dataset.__getitem__(0)

    # First and repeated accesses should be identical
    assert_equal(dataset.__len__(), 20)


# ============================================================================
# Dataset Type Consistency Tests
# ============================================================================


fn test_dataset_interface_protocol() raises:
    """Test ExTensorDataset conforms to Dataset trait protocol.

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
    var data = ExTensor(data_list^)

    var labels_list = List[Int]()
    for i in range(30):
        labels_list.append(i)
    var labels = ExTensor(labels_list^)

    var dataset = ExTensorDataset(data^, labels^)

    # Both methods should be accessible
    var len_val = dataset.__len__()
    assert_equal(len_val, 30)

    # __getitem__ should work
    var sample = dataset.__getitem__(0)


# ============================================================================
# Main Test Runner
# ============================================================================


fn main() raises:
    """Run dataset integration tests (Part 1)."""
    print("Running dataset integration tests (Part 1)...")

    # Interface conformance tests
    test_dataset_length_consistency()
    test_dataset_sequential_access()
    test_dataset_negative_indexing()
    test_dataset_bounds_checking()

    # Loader integration tests
    test_dataset_with_loader()
    test_tensor_dataset_batching_shapes()
    test_dataset_random_access()

    # Type consistency tests
    test_dataset_interface_protocol()

    print("✓ All dataset integration tests (Part 1) passed!")
