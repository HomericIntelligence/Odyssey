"""Tests for TransformedDataset wrapper.

Tests that TransformedDataset correctly applies transforms to data
while leaving labels unchanged.
"""

from tests.shared.conftest import (
    assert_true,
    assert_equal,
    TestFixtures,
)
from shared.data import AnyTensorDataset, TransformedDataset
from shared.data.transforms import Normalize
from shared.core.any_tensor import AnyTensor, ones, zeros
from collections import List


# ============================================================================
# TransformedDataset Creation Tests
# ============================================================================


fn test_transformed_dataset_creation() raises:
    """Test creating TransformedDataset.

    TransformedDataset should wrap a base dataset and a transform.
    """
    var data_shape: List[Int] = [10, 3, 32, 32]
    var label_shape: List[Int] = [10, 10]

    var data = ones(data_shape, DType.float32)
    var labels = zeros(label_shape, DType.float32)

    var base_dataset = AnyTensorDataset(data^, labels^)
    var normalize = Normalize(mean=0.5, std=0.5)
    var transformed = TransformedDataset(base_dataset^, normalize^)

    assert_equal(transformed.__len__(), 10)


fn test_transformed_dataset_length() raises:
    """Test that TransformedDataset.__len__ matches base dataset.

    The length should reflect the number of samples in the base dataset.
    """
    var data_shape: List[Int] = [42, 3, 32, 32]
    var label_shape: List[Int] = [42, 10]

    var data = ones(data_shape, DType.float32)
    var labels = zeros(label_shape, DType.float32)

    var base_dataset = AnyTensorDataset(data^, labels^)
    var normalize = Normalize(mean=0.5, std=0.5)
    var transformed = TransformedDataset(base_dataset^, normalize^)

    assert_equal(transformed.__len__(), 42)


fn test_transformed_dataset_applies_transform() raises:
    """Test that transform is applied to data.

    The transform should modify the data tensor but not the labels.
    """
    var data_shape: List[Int] = [5, 1, 8, 8]
    var label_shape: List[Int] = [5, 10]

    var data = ones(data_shape, DType.float32)
    var labels = zeros(label_shape, DType.float32)

    var base_dataset = AnyTensorDataset(data^, labels^)
    var normalize = Normalize(mean=0.5, std=0.5)
    var transformed = TransformedDataset(base_dataset^, normalize^)

    # Get a sample with transform
    var transformed_data, transformed_labels = transformed[0]

    # Original data was all 1.0, normalize with mean=0.5, std=0.5
    # should give (1.0 - 0.5) / 0.5 = 1.0
    # Note: Original data was 1.0, so transformed should also be 1.0
    var transformed_first = transformed_data._data.bitcast[Float32]()[0]

    # Transformed value should be 1.0 (from normalize: (1.0 - 0.5) / 0.5 = 1.0)
    assert_true(transformed_first > 0.9)
    assert_true(transformed_first < 1.1)


fn test_transformed_dataset_preserves_labels() raises:
    """Test that labels are not transformed.

    The transform should only affect data, labels should pass through unchanged.
    """
    var data_shape: List[Int] = [3, 1, 8, 8]
    var label_shape: List[Int] = [3, 10]

    var data = ones(data_shape, DType.float32)
    var labels = zeros(label_shape, DType.float32)

    var base_dataset = AnyTensorDataset(data^, labels^)
    var normalize = Normalize(mean=0.5, std=0.5)
    var transformed = TransformedDataset(base_dataset^, normalize^)

    var _, transformed_labels = transformed[0]

    # Labels should still be zeros (unchanged from original)
    # Original labels were created with zeros(), so first element should be 0.0
    var trans_first = transformed_labels._data.bitcast[Float32]()[0]

    assert_equal(trans_first, Float32(0.0))


fn test_transformed_dataset_all_samples() raises:
    """Test that transform is applied to all samples consistently.

    Each sample should have the transform applied when accessed.
    """
    var data_shape: List[Int] = [10, 1, 4, 4]
    var label_shape: List[Int] = [10, 10]

    var data = ones(data_shape, DType.float32)
    var labels = zeros(label_shape, DType.float32)

    var base_dataset = AnyTensorDataset(data^, labels^)
    var normalize = Normalize(mean=0.5, std=0.5)
    var transformed = TransformedDataset(base_dataset^, normalize^)

    # Check a few samples
    for i in range(3):
        var trans_data, _ = transformed[i]
        var trans_first = trans_data._data.bitcast[Float32]()[0]

        # All should have same normalized value
        assert_true(trans_first > 0.9)
        assert_true(trans_first < 1.1)


fn main() raises:
    """Run all tests."""
    print("Testing TransformedDataset...")
    test_transformed_dataset_creation()
    test_transformed_dataset_length()
    test_transformed_dataset_applies_transform()
    test_transformed_dataset_preserves_labels()
    test_transformed_dataset_all_samples()
    print("All TransformedDataset tests passed!")
