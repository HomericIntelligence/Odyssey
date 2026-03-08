"""Unit tests for data constants module (part 2 of 2).

Tests for:
    - DatasetInfo struct with EMNIST variants
    - DatasetInfo error handling
    - DatasetInfo image shape across all datasets
    - Class name utility functions

# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_constants.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""

from testing import assert_equal, assert_true
from shared.data.constants import (
    CIFAR10_CLASS_NAMES,
    EMNIST_BALANCED_CLASSES,
    EMNIST_BYCLASS_CLASSES,
    EMNIST_BYMERGE_CLASSES,
    EMNIST_DIGITS_CLASSES,
    EMNIST_LETTERS_CLASSES,
    DatasetInfo,
)


fn test_dataset_info_emnist_byclass() raises:
    """Test DatasetInfo with EMNIST By Class."""
    var info = DatasetInfo("emnist_byclass")

    assert_equal(info.num_classes(), 62, "EMNIST By Class has 62 classes")
    assert_equal(
        info.num_train_samples(),
        814255,
        "EMNIST By Class has ~814255 training samples",
    )
    assert_equal(
        info.num_test_samples(),
        135800,
        "EMNIST By Class has ~135800 test samples",
    )

    var shape = info.image_shape()
    assert_equal(shape[0], 1, "EMNIST is grayscale")
    assert_equal(shape[1], 28, "EMNIST images are 28x28")
    assert_equal(shape[2], 28, "EMNIST images are 28x28")


fn test_dataset_info_emnist_bymerge() raises:
    """Test DatasetInfo with EMNIST By Merge."""
    var info = DatasetInfo("emnist_bymerge")

    assert_equal(info.num_classes(), 36, "EMNIST By Merge has 36 classes")


fn test_dataset_info_emnist_digits() raises:
    """Test DatasetInfo with EMNIST Digits."""
    var info = DatasetInfo("emnist_digits")

    assert_equal(info.num_classes(), 10, "EMNIST Digits has 10 classes")
    assert_equal(
        info.num_train_samples(),
        60000,
        "EMNIST Digits has 60000 training samples",
    )
    assert_equal(
        info.num_test_samples(), 10000, "EMNIST Digits has 10000 test samples"
    )


fn test_dataset_info_emnist_letters() raises:
    """Test DatasetInfo with EMNIST Letters."""
    var info = DatasetInfo("emnist_letters")

    assert_equal(info.num_classes(), 52, "EMNIST Letters has 52 classes")
    assert_equal(
        info.num_train_samples(),
        103600,
        "EMNIST Letters has ~103600 training samples",
    )
    assert_equal(
        info.num_test_samples(), 17383, "EMNIST Letters has ~17383 test samples"
    )


fn test_dataset_info_invalid_dataset() raises:
    """Test DatasetInfo with invalid dataset name."""
    var error_raised = False
    try:
        var info = DatasetInfo("invalid_dataset")
    except:
        error_raised = True

    assert_true(error_raised, "Invalid dataset should raise error")


fn test_dataset_info_class_name_out_of_range() raises:
    """Test DatasetInfo.class_name() with out-of-range index."""
    var info = DatasetInfo("cifar10")

    # Negative index should raise error
    var neg_error = False
    try:
        var _ = info.class_name(-1)
    except:
        neg_error = True
    assert_true(neg_error, "Negative class index should raise error")

    # Out of range index should raise error
    var range_error = False
    try:
        var _ = info.class_name(100)
    except:
        range_error = True
    assert_true(range_error, "Out-of-range class index should raise error")


fn test_dataset_info_image_shape_all_datasets() raises:
    """Test image_shape for all datasets."""
    var datasets = List[String]()
    datasets.append("cifar10")
    datasets.append("emnist_balanced")
    datasets.append("emnist_byclass")
    datasets.append("emnist_bymerge")
    datasets.append("emnist_digits")
    datasets.append("emnist_letters")

    for dataset_name in datasets:
        var info = DatasetInfo(dataset_name)
        var shape = info.image_shape()

        assert_equal(
            len(shape),
            3,
            "Image shape should have 3 dimensions for " + dataset_name,
        )

        if dataset_name == "cifar10":
            assert_equal(shape[0], 3, "CIFAR-10 should have 3 channels")
            assert_equal(shape[1], 32, "CIFAR-10 height should be 32")
            assert_equal(shape[2], 32, "CIFAR-10 width should be 32")
        else:
            # All EMNIST variants
            assert_equal(
                shape[0], 1, "EMNIST should have 1 channel for " + dataset_name
            )
            assert_equal(
                shape[1], 28, "EMNIST height should be 28 for " + dataset_name
            )
            assert_equal(
                shape[2], 28, "EMNIST width should be 28 for " + dataset_name
            )


fn test_class_names_not_empty() raises:
    """Test that all class name functions return non-empty lists."""
    assert_true(
        len(CIFAR10_CLASS_NAMES()) > 0, "CIFAR10 classes should not be empty"
    )
    assert_true(
        len(EMNIST_BALANCED_CLASSES()) > 0,
        "EMNIST Balanced classes should not be empty",
    )
    assert_true(
        len(EMNIST_BYCLASS_CLASSES()) > 0,
        "EMNIST By Class classes should not be empty",
    )
    assert_true(
        len(EMNIST_BYMERGE_CLASSES()) > 0,
        "EMNIST By Merge classes should not be empty",
    )
    assert_true(
        len(EMNIST_DIGITS_CLASSES()) > 0,
        "EMNIST Digits classes should not be empty",
    )
    assert_true(
        len(EMNIST_LETTERS_CLASSES()) > 0,
        "EMNIST Letters classes should not be empty",
    )


fn main() raises:
    """Run all tests."""
    print("Testing DatasetInfo with EMNIST By Class...")
    test_dataset_info_emnist_byclass()
    print("  PASSED")

    print("Testing DatasetInfo with EMNIST By Merge...")
    test_dataset_info_emnist_bymerge()
    print("  PASSED")

    print("Testing DatasetInfo with EMNIST Digits...")
    test_dataset_info_emnist_digits()
    print("  PASSED")

    print("Testing DatasetInfo with EMNIST Letters...")
    test_dataset_info_emnist_letters()
    print("  PASSED")

    print("Testing DatasetInfo with invalid dataset...")
    test_dataset_info_invalid_dataset()
    print("  PASSED")

    print("Testing DatasetInfo.class_name() with out-of-range index...")
    test_dataset_info_class_name_out_of_range()
    print("  PASSED")

    print("Testing image_shape for all datasets...")
    test_dataset_info_image_shape_all_datasets()
    print("  PASSED")

    print("Testing class names are not empty...")
    test_class_names_not_empty()
    print("  PASSED")

    print("\nAll tests passed!")
