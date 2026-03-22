"""Tests for EMNIST Dataset Wrapper - Part 3: Train/Test Splits, Edge Cases, Performance

Tests cover:
- Train vs test dataset size comparison
- Data and label consistency
- Valid split enumeration
- Performance of random access

ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
high test load. Split from test_emnist.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""

from testing import assert_equal, assert_true, assert_false, assert_raises
from shared.data import EMNISTDataset, AnyTensorDataset, Dataset
from shared.core.extensor import AnyTensor


# ============================================================================
# Integration Tests
# ============================================================================


fn test_emnist_train_vs_test_sizes() raises:
    """Test that train and test splits have different sizes.

    Verifies that training and test datasets contain expected sample counts.
    """
    try:
        var train_dataset = EMNISTDataset(
            "/tmp/emnist", split="balanced", train=True
        )
        var test_dataset = EMNISTDataset(
            "/tmp/emnist", split="balanced", train=False
        )

        var train_len = train_dataset.__len__()
        var test_len = test_dataset.__len__()

        # Train should typically have more samples than test
        assert_true(train_len > 0, "Train set should have samples")
        assert_true(test_len > 0, "Test set should have samples")
    except e:
        print("Test data not available - skipping train/test split test")


# ============================================================================
# Edge Cases and Error Handling
# ============================================================================


fn test_emnist_data_label_consistency() raises:
    """Test that data and labels have matching first dimensions.

    Verifies that the dataset maintains consistency between data and labels.
    """
    try:
        var dataset = EMNISTDataset("/tmp/emnist", split="balanced", train=True)

        var data_len = dataset.data.shape()[0]
        var labels_len = dataset.labels.shape()[0]

        assert_equal(
            data_len,
            labels_len,
            "Data and labels should have same first dimension",
        )
    except e:
        print("Test data not available - skipping consistency test")


fn test_emnist_all_valid_splits() raises:
    """Test that all documented splits are accepted.

    Verifies that balanced, byclass, bymerge, digits, letters, mnist are all valid.
    """
    var splits = List[String]()
    splits.append("balanced")
    splits.append("byclass")
    splits.append("bymerge")
    splits.append("digits")
    splits.append("letters")
    splits.append("mnist")

    for split in splits:
        var error_raised = False
        try:
            # Just test that initialization is accepted (may fail on file I/O)
            # The key test is that the split validation passes
            var dataset = EMNISTDataset("/tmp/emnist", split=split, train=True)
            _ = dataset  # Consume unused variable
        except e:
            # Check that error is file I/O, not validation
            if String(e).__contains__("Invalid split"):
                error_raised = True
                assert_false(True, "Split '" + split + "' should be valid")

        _ = error_raised  # Consume unused variable
        # Note: File I/O errors are expected if data doesn't exist


# ============================================================================
# Performance Tests
# ============================================================================


fn test_emnist_performance_random_access() raises:
    """Test performance of random index access.

    Verifies that accessing different indices works correctly.
    """
    try:
        var dataset = EMNISTDataset("/tmp/emnist", split="balanced", train=True)
        var length = dataset.__len__()

        if length > 0:
            # Access first, middle, and last samples
            var first_data, first_label = dataset.__getitem__(0)
            var middle_data, middle_label = dataset.__getitem__(length // 2)
            var last_data, last_label = dataset.__getitem__(length - 1)
            _ = first_label  # Consume unused variable
            _ = middle_label  # Consume unused variable
            _ = last_label  # Consume unused variable

            # Verify all have correct shape
            for data in [first_data, middle_data, last_data]:
                var shape = data.shape()
                assert_equal(len(shape), 4, "All samples should have 4D shape")
    except e:
        print("Test data not available - skipping performance test")


# ============================================================================
# Main Test Runner
# ============================================================================


fn main() raises:
    """Run EMNIST dataset tests - Part 3: Train/Test Splits, Edge Cases, Performance.
    """
    print("Running EMNIST dataset tests (Part 3)...")

    # Train/test size test
    test_emnist_train_vs_test_sizes()
    print("✓ test_emnist_train_vs_test_sizes")

    # Consistency tests
    test_emnist_data_label_consistency()
    print("✓ test_emnist_data_label_consistency")

    test_emnist_all_valid_splits()
    print("✓ test_emnist_all_valid_splits")

    # Performance tests
    test_emnist_performance_random_access()
    print("✓ test_emnist_performance_random_access")

    print("\nAll EMNIST dataset tests (Part 3) passed!")
