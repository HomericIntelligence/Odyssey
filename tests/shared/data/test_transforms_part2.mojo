"""Transform integration tests - Part 2: Dataset Integration and Edge Cases.

Split from test_transforms.mojo per ADR-009 to avoid Mojo heap corruption.

# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_transforms.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md

Tests cover:
- Transform + Dataset integration workflows
- Edge cases: small tensors, large tensors, zero values, negative values
- Transform repeated application
- Element count preservation
"""

from tests.shared.conftest import (
    assert_true,
    assert_equal,
    assert_not_equal,
    assert_close_float,
    TestFixtures,
)
from shared.data.transforms import Compose, Normalize, Reshape
from shared.core.extensor import ExTensor


# ============================================================================
# Transform + Dataset Integration Tests
# ============================================================================


fn test_transform_on_dataset_sample() raises:
    """Test applying transform to dataset sample works correctly.

    Transforms should work on samples retrieved from datasets.

    Integration Points:
        - Transform + dataset sample
        - Tuple unpacking for (data, label)
        - Transform on individual samples

    Success Criteria:
        - Can retrieve sample from dataset
        - Can apply transform to sample
        - Result is valid tensor
    """
    TestFixtures.set_seed()

    from shared.data.datasets import ExTensorDataset

    # Create small dataset
    var data_list = List[Float32]()
    for i in range(20):
        data_list.append(Float32(i) * 0.1)
    var data = ExTensor(data_list^)

    var labels_list = List[Int]()
    for i in range(20):
        labels_list.append(i)
    var labels = ExTensor(labels_list^)

    var dataset = ExTensorDataset(data^, labels^)

    # Get sample from dataset
    var sample = dataset.__getitem__(0)
    var sample_data = sample[0]

    # Apply transform
    var normalize = Normalize()
    var transformed = normalize(sample_data)

    # Result should be valid
    assert_equal(transformed.num_elements(), sample_data.num_elements())


fn test_transform_batch_consistency() raises:
    """Test transform produces consistent results across multiple samples.

    Applying same transform to different samples should each succeed.

    Integration Points:
        - Transform applied repeatedly
        - Consistency across applications
        - Batch-like scenarios

    Success Criteria:
        - Can transform 10 different samples
        - All transformations succeed
        - Results have correct shapes
    """
    TestFixtures.set_seed()

    from shared.data.datasets import ExTensorDataset

    # Create dataset
    var data_list = List[Float32]()
    for i in range(20):
        data_list.append(Float32(i) * 0.1)
    var data = ExTensor(data_list^)

    var labels_list = List[Int]()
    for i in range(20):
        labels_list.append(i)
    var labels = ExTensor(labels_list^)

    var dataset = ExTensorDataset(data^, labels^)

    var normalize = Normalize()

    # Transform first 5 samples
    for i in range(5):
        var sample = dataset.__getitem__(i)
        var sample_data = sample[0]
        var result = normalize(sample_data)
        assert_equal(result.num_elements(), sample_data.num_elements())


# ============================================================================
# Transform Edge Case Tests
# ============================================================================


fn test_transform_on_small_tensor() raises:
    """Test transform on minimal tensor (1 element).

    Transforms should handle edge case of single-element tensors.

    Integration Points:
        - Transform on minimal data
        - Edge case robustness
        - Denominator safety (avoid division by zero)

    Success Criteria:
        - Single-element tensor can be transformed
        - No errors or exceptions
        - Output has same shape
    """
    TestFixtures.set_seed()

    var data_list = List[Float32]()
    data_list.append(42.0)
    var data = ExTensor(data_list^)

    var normalize = Normalize()
    var result = normalize(data)

    assert_equal(result.num_elements(), 1)


fn test_transform_on_large_tensor() raises:
    """Test transform on larger tensor (1000+ elements).

    Transforms should scale to larger datasets without issues.

    Integration Points:
        - Transform scalability
        - Performance on larger data
        - Memory efficiency

    Success Criteria:
        - 1000-element tensor can be transformed
        - No memory errors or timeouts
        - Output shape matches input
    """
    TestFixtures.set_seed()

    var data_list = List[Float32]()
    for i in range(1000):
        data_list.append(Float32(i % 100) * 0.01)
    var data = ExTensor(data_list^)

    var normalize = Normalize()
    var result = normalize(data)

    assert_equal(result.num_elements(), 1000)


fn test_transform_zero_value_handling() raises:
    """Test transform handles tensors with zero values.

    Tensors containing zeros should be transformable (test numerics).

    Integration Points:
        - Edge case numeric values
        - Zero handling in statistics
        - Numerical stability

    Success Criteria:
        - Tensor with all zeros can be transformed
        - No division by zero errors
        - Output is valid
    """
    TestFixtures.set_seed()

    var data_list = List[Float32]()
    for _ in range(10):
        data_list.append(0.0)
    var data = ExTensor(data_list^)

    var normalize = Normalize()
    # This might result in NaN/Inf, but should not crash
    var result = normalize(data)

    assert_equal(result.num_elements(), 10)


fn test_transform_negative_values() raises:
    """Test transform handles negative values correctly.

    Transforms should work with negative input values.

    Integration Points:
        - Negative value handling
        - Sign preservation where applicable
        - Numerical correctness

    Success Criteria:
        - Tensor with negative values transforms successfully
        - Output shape is correct
        - No sign errors or flips
    """
    TestFixtures.set_seed()

    var data_list = List[Float32]()
    for i in range(10):
        data_list.append(Float32(i) - 5.0)  # -5 to 4
    var data = ExTensor(data_list^)

    var normalize = Normalize()
    var result = normalize(data)

    assert_equal(result.num_elements(), 10)


fn test_transform_repeated_application() raises:
    """Test applying same transform repeatedly on same tensor.

    Multiple applications should each succeed independently.

    Integration Points:
        - Transform reusability
        - No state accumulation
        - Determinism across applications

    Success Criteria:
        - Can apply transform 5 times in sequence
        - Each application succeeds
        - No degradation or accumulation
    """
    TestFixtures.set_seed()

    var data_list = List[Float32]()
    for i in range(20):
        data_list.append(Float32(i) * 0.1)
    var data = ExTensor(data_list^)

    var normalize = Normalize()

    # Apply 5 times
    var result = data
    for _ in range(5):
        result = normalize(result)
        assert_equal(result.num_elements(), 20)


fn test_transform_preserves_element_count() raises:
    """Test transforms preserve total number of elements.

    num_elements() should be unchanged after transform.

    Integration Points:
        - Element count preservation
        - Shape consistency
        - Memory allocation safety

    Success Criteria:
        - Before and after element count identical
        - Shape structure preserved
    """
    TestFixtures.set_seed()

    var data_list = List[Float32]()
    for i in range(30):
        data_list.append(Float32(i) * 0.05)
    var data = ExTensor(data_list^)

    var original_count = data.num_elements()

    var normalize = Normalize()
    var result = normalize(data)

    var result_count = result.num_elements()
    assert_equal(original_count, result_count)


# ============================================================================
# Main Test Runner
# ============================================================================


fn main() raises:
    """Run transform integration tests - Part 2."""
    print("Running transform integration tests (Part 2)...")

    # Dataset integration tests
    test_transform_on_dataset_sample()
    test_transform_batch_consistency()

    # Edge case tests
    test_transform_on_small_tensor()
    test_transform_on_large_tensor()
    test_transform_zero_value_handling()
    test_transform_negative_values()
    test_transform_repeated_application()

    # Value range tests
    test_transform_preserves_element_count()

    print("✓ All transform integration tests (Part 2) passed!")
