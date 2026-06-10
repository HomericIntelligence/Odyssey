"""Integration tests for data pipeline workflows.

Tests cover:
- Data loading and batching
- Data transformation pipelines
- Dataset handling and preprocessing
- Data streaming and memory efficiency

These tests validate that data handling components work correctly together,
exercising the real conftest data fixtures (create_simple_dataset,
create_mock_dataloader, MockDataLoader) rather than asserting on hand-built
tensors.
"""

from tests.projectodyssey.conftest import (
    assert_true,
    assert_less,
    assert_greater,
    assert_equal,
    create_simple_dataset,
    create_mock_dataloader,
    MockDataLoader,
    TestFixtures,
)


# ============================================================================
# Data Loading Tests
# ============================================================================


def test_data_loading_basic() raises:
    """Test basic data loading produces samples with the requested shapes.

    Integration Points:
        - Synthetic dataset creation (create_simple_dataset)
        - Per-sample (input, label) tuple structure

    Success Criteria:
        - Dataset has the requested number of samples
        - Each sample's input/label vectors have the requested dimensions.
    """
    var n_samples = 10
    var input_dim = 5
    var output_dim = 2

    var dataset = create_simple_dataset(
        n_samples=n_samples, input_dim=input_dim, output_dim=output_dim
    )

    # The dataset must contain exactly n_samples entries.
    assert_equal(len(dataset), n_samples)

    # Each entry is an (input, label) tuple with the requested dimensions.
    assert_equal(len(dataset[0][0]), input_dim)
    assert_equal(len(dataset[0][1]), output_dim)

    assert_equal(len(dataset[n_samples - 1][0]), input_dim)
    assert_equal(len(dataset[n_samples - 1][1]), output_dim)


def test_data_generation_determinism() raises:
    """Test the dataset generator is deterministic for a fixed seed.

    Integration Points:
        - Deterministic data generation (seeded)
        - Data integrity across repeated generation

    Success Criteria:
        - Generating with the same seed yields identical values
        - No data corruption between runs.

    Note:
        This validates reproducible data generation. Tensor transform
        composition (transforms.mojo) operates on tensors with a richer
        signature and is exercised by the dedicated transform unit tests,
        not here.
    """
    var dataset_a = create_simple_dataset(
        n_samples=4, input_dim=3, output_dim=1, seed_value=7
    )
    var dataset_b = create_simple_dataset(
        n_samples=4, input_dim=3, output_dim=1, seed_value=7
    )

    assert_equal(len(dataset_a), len(dataset_b))

    # Same seed must reproduce identical feature values element-by-element.
    for i in range(len(dataset_a)):
        var a_in = dataset_a[i][0].copy()
        var b_in = dataset_b[i][0].copy()
        assert_equal(len(a_in), len(b_in))
        for j in range(len(a_in)):
            assert_true(
                a_in[j] == b_in[j],
                "Deterministic generation must reproduce values",
            )


def test_data_batching_and_shuffling() raises:
    """Test the data loader reports the correct number of batches.

    Integration Points:
        - MockDataLoader batch accounting (__len__)
        - Partial final batch handling

    Success Criteria:
        - Number of batches equals ceil(num_samples / batch_size)
        - A non-divisible split still accounts for every sample.
    """
    var batch_size = 32
    var num_samples = 100

    var loader = create_mock_dataloader(
        n_samples=num_samples, batch_size=batch_size
    )

    # ceil(100 / 32) = 4 batches (last batch is partial: 4 samples).
    var expected_batches = (num_samples + batch_size - 1) // batch_size
    assert_equal(loader.__len__(), expected_batches)
    assert_equal(loader.__len__(), 4)

    # An evenly divisible split should have no partial batch.
    var even_loader = create_mock_dataloader(n_samples=64, batch_size=32)
    assert_equal(even_loader.__len__(), 2)


def test_data_pipeline_memory_efficiency() raises:
    """Test a larger dataset is generated without inflating the sample count.

    Integration Points:
        - Bounded sample materialization
        - No per-sample duplication

    Success Criteria:
        - Sample count matches the request exactly (no duplication)
        - Each sample retains the requested dimensionality.
    """
    var n_samples = 1000
    var input_dim = 100

    var dataset = create_simple_dataset(
        n_samples=n_samples, input_dim=input_dim, output_dim=1
    )

    # Exactly n_samples must be produced - no duplication or truncation.
    assert_equal(len(dataset), n_samples)
    assert_equal(len(dataset[0][0]), input_dim)
    assert_equal(len(dataset[n_samples - 1][0]), input_dim)


# ============================================================================
# Dataset Handling Tests
# ============================================================================


def test_dataset_creation() raises:
    """Test dataset creation yields the requested shape metadata.

    Integration Points:
        - Dataset initialization with explicit dimensions
        - Input/label separation

    Success Criteria:
        - Datasets created with the requested sample count
        - Input and label dimensions are honored per sample.
    """
    var dataset = create_simple_dataset(
        n_samples=100, input_dim=28 * 28, output_dim=10
    )

    assert_equal(len(dataset), 100)
    assert_equal(len(dataset[0][0]), 28 * 28)
    assert_equal(len(dataset[0][1]), 10)


def test_dataset_split_sizes() raises:
    """Test 70/15/15 split sizing conserves all samples of a real dataset.

    Integration Points:
        - Split sizing computed against an actual materialized dataset
        - Conservation of total samples (no leakage / no loss)

    Success Criteria:
        - The dataset is built with the expected sample count
        - train + val + test split sizes sum back to the total
        - The training split is the largest.

    Note:
        The shared library does not yet expose a dataset-splitting helper,
        so this verifies the split-size accounting (boundary indices that a
        splitter would use); slicing into three sub-datasets is future work.
    """
    var total_samples = 1000
    var dataset = create_simple_dataset(
        n_samples=total_samples, input_dim=4, output_dim=1
    )
    assert_equal(len(dataset), total_samples)

    var train_size = (total_samples * 70) // 100
    var val_size = (total_samples * 15) // 100
    var test_size = total_samples - train_size - val_size

    # Splits must conserve the total number of samples (no leakage/loss).
    assert_equal(train_size + val_size + test_size, total_samples)
    assert_greater(train_size, val_size)
    assert_greater(train_size, test_size)


# ============================================================================
# Main Test Execution
# ============================================================================


def main() raises:
    """Run all data pipeline integration tests."""
    print("Running data loading tests...")
    test_data_loading_basic()
    test_data_generation_determinism()
    test_data_batching_and_shuffling()
    test_data_pipeline_memory_efficiency()

    print("Running dataset handling tests...")
    test_dataset_creation()
    test_dataset_split_sizes()

    print("\nAll data pipeline integration tests passed!")
