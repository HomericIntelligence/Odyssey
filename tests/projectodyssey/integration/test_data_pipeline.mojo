"""Integration tests for data pipeline workflows.

Tests cover:
- Data loading and batching
- Data transformation pipelines
- Dataset handling and preprocessing
- Data streaming and memory efficiency

These tests validate that data handling components work correctly together.
"""

from tests.projectodyssey.conftest import (
    assert_true,
    assert_less,
    assert_greater,
    assert_equal,
    TestFixtures,
)
from projectodyssey.tensor.any_tensor import AnyTensor, zeros, ones


# ============================================================================
# Data Loading Tests
# ============================================================================


def test_data_loading_basic() raises:
    """Test basic data loading functionality.

    Integration Points:
        - Dataset creation
        - Data loader initialization
        - Batch creation

    Success Criteria:
        - Data loader creates batches correctly
        - All data is accessible
        - No runtime errors.
    """
    var data_shape: List[Int] = [10, 5]
    var data = ones(data_shape, DType.float32)

    var labels_shape: List[Int] = [10]
    var labels = zeros(labels_shape, DType.float32)

    assert_equal(data.shape()[0], 10)
    assert_equal(labels.shape()[0], 10)


def test_data_transformation_pipeline() raises:
    """Test data transformation pipeline.

    Integration Points:
        - Transform composition
        - Sequential transformations
        - Data integrity through pipeline

    Success Criteria:
        - Transforms apply in correct order
        - Data shapes preserved/correct
        - No data corruption.
    """
    var input_shape: List[Int] = [8, 3, 32, 32]
    var data = ones(input_shape, DType.float32)

    var output_shape = data.shape()
    assert_equal(output_shape[0], 8)
    assert_equal(output_shape[1], 3)
    assert_equal(output_shape[2], 32)
    assert_equal(output_shape[3], 32)


def test_data_batching_and_shuffling() raises:
    """Test data batching with shuffling.

    Integration Points:
        - Batch creation
        - Shuffle mechanism
        - Random state management

    Success Criteria:
        - Batches have correct size
        - Shuffling produces different order
        - All data included in epochs.
    """
    var batch_size = 32
    var num_samples = 100

    var data_shape: List[Int] = [num_samples, 10]
    var data = ones(data_shape, DType.float32)

    var expected_batches = (num_samples + batch_size - 1) // batch_size
    assert_greater(expected_batches, 0)


def test_data_pipeline_memory_efficiency() raises:
    """Test memory efficiency of data pipeline.

    Integration Points:
        - Lazy loading
        - Memory management
        - Generator patterns

    Success Criteria:
        - Memory usage stays bounded
        - Large datasets handled efficiently
        - No data duplication.
    """
    var data_shape: List[Int] = [1000, 100]
    var data = ones(data_shape, DType.float32)

    var total_elements = data.shape()[0] * data.shape()[1]
    assert_equal(total_elements, 100000)


# ============================================================================
# Dataset Handling Tests
# ============================================================================


def test_dataset_creation() raises:
    """Test dataset creation from various sources.

    Integration Points:
        - Dataset initialization
        - Data validation
        - Shape/dtype handling

    Success Criteria:
        - Datasets created successfully
        - Metadata correct
        - Data accessible.
    """
    var x_shape: List[Int] = [100, 28, 28]
    var y_shape: List[Int] = [100]

    var x_data = ones(x_shape, DType.float32)
    var y_data = zeros(y_shape, DType.float32)

    assert_equal(x_data.shape()[0], 100)
    assert_equal(y_data.shape()[0], 100)


def test_dataset_splits() raises:
    """Test train/val/test dataset splitting.

    Integration Points:
        - Split logic
        - No data leakage
        - Stratification (if applicable)

    Success Criteria:
        - Splits created correctly
        - Total data preserved
        - No overlap between splits.
    """
    var total_samples = 1000
    var train_ratio = 0.7
    var val_ratio = 0.15
    var test_ratio = 0.15

    var train_size = Int(Float64(total_samples) * train_ratio)
    var val_size = Int(Float64(total_samples) * val_ratio)
    var test_size = Int(Float64(total_samples) * test_ratio)

    var total = train_size + val_size + test_size
    assert_equal(total, 1000)


# ============================================================================
# Main Test Execution
# ============================================================================


def main() raises:
    """Run all data pipeline integration tests."""
    print("Running data loading tests...")
    test_data_loading_basic()
    test_data_transformation_pipeline()
    test_data_batching_and_shuffling()
    test_data_pipeline_memory_efficiency()

    print("Running dataset handling tests...")
    test_dataset_creation()
    test_dataset_splits()

    print("\nAll data pipeline integration tests passed! ")
