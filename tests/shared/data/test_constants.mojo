"""Unit tests for data constants module (part 1 of 2).

Tests for:
    - CIFAR-10 class names
    - EMNIST class names (all splits)
    - DatasetInfo struct with CIFAR-10 and EMNIST Balanced

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


fn test_cifar10_class_names() raises:
    """Test CIFAR-10 class names are correct."""
    var classes = CIFAR10_CLASS_NAMES()
    assert_equal(len(classes), 10, "CIFAR-10 should have 10 classes")

    # Test specific class names
    assert_equal(classes[0], "airplane", "Class 0 should be airplane")
    assert_equal(classes[1], "automobile", "Class 1 should be automobile")
    assert_equal(classes[2], "bird", "Class 2 should be bird")
    assert_equal(classes[3], "cat", "Class 3 should be cat")
    assert_equal(classes[4], "deer", "Class 4 should be deer")
    assert_equal(classes[5], "dog", "Class 5 should be dog")
    assert_equal(classes[6], "frog", "Class 6 should be frog")
    assert_equal(classes[7], "horse", "Class 7 should be horse")
    assert_equal(classes[8], "ship", "Class 8 should be ship")
    assert_equal(classes[9], "truck", "Class 9 should be truck")


fn test_emnist_balanced_classes() raises:
    """Test EMNIST Balanced class names."""
    var classes = EMNIST_BALANCED_CLASSES()
    assert_equal(len(classes), 47, "EMNIST Balanced should have 47 classes")

    # Test digit classes (0-9)
    for i in range(10):
        var expected = String(i)
        assert_equal(classes[i], expected, "Digit class index " + String(i))

    # Test uppercase letters (A-Z) - indices 10-35
    for i in range(26):
        var expected = chr(ord("A") + i)
        assert_equal(
            classes[10 + i],
            expected,
            "Uppercase letter at index " + String(10 + i),
        )

    # EMNIST Balanced has only 11 lowercase letters at indices 36-46
    # These are letters that look different from uppercase: a, b, d, e, f, g, h, n, q, r, t
    # Total: 10 digits + 26 uppercase + 11 lowercase = 47 classes
    assert_equal(len(classes), 47, "Total should be 47 classes")


fn test_emnist_byclass_classes() raises:
    """Test EMNIST By Class class names."""
    var classes = EMNIST_BYCLASS_CLASSES()
    assert_equal(len(classes), 62, "EMNIST By Class should have 62 classes")

    # Test digit classes (0-9)
    for i in range(10):
        var expected = String(i)
        assert_equal(classes[i], expected, "Digit class index " + String(i))

    # Test uppercase letters (A-Z) - indices 10-35
    for i in range(26):
        var expected = chr(ord("A") + i)
        assert_equal(
            classes[10 + i],
            expected,
            "Uppercase letter at index " + String(10 + i),
        )

    # Test lowercase letters (a-z) - indices 36-61
    for i in range(26):
        var expected = chr(ord("a") + i)
        assert_equal(
            classes[36 + i],
            expected,
            "Lowercase letter at index " + String(36 + i),
        )


fn test_emnist_bymerge_classes() raises:
    """Test EMNIST By Merge class names."""
    var classes = EMNIST_BYMERGE_CLASSES()
    assert_equal(len(classes), 36, "EMNIST By Merge should have 36 classes")

    # Test digit classes (0-9)
    for i in range(10):
        var expected = String(i)
        assert_equal(classes[i], expected, "Digit class index " + String(i))

    # Test merged letters (A-Z only) - indices 10-35
    for i in range(26):
        var expected = chr(ord("A") + i)
        assert_equal(
            classes[10 + i],
            expected,
            "Merged letter at index " + String(10 + i),
        )


fn test_emnist_digits_classes() raises:
    """Test EMNIST Digits class names."""
    var classes = EMNIST_DIGITS_CLASSES()
    assert_equal(len(classes), 10, "EMNIST Digits should have 10 classes")

    # Test digit classes (0-9)
    for i in range(10):
        var expected = String(i)
        assert_equal(classes[i], expected, "Digit class " + String(i))


fn test_emnist_letters_classes() raises:
    """Test EMNIST Letters class names."""
    var classes = EMNIST_LETTERS_CLASSES()
    assert_equal(len(classes), 52, "EMNIST Letters should have 52 classes")

    # Test uppercase letters (A-Z) - indices 0-25
    for i in range(26):
        var expected = chr(ord("A") + i)
        assert_equal(
            classes[i], expected, "Uppercase letter at index " + String(i)
        )

    # Test lowercase letters (a-z) - indices 26-51
    for i in range(26):
        var expected = chr(ord("a") + i)
        assert_equal(
            classes[26 + i],
            expected,
            "Lowercase letter at index " + String(26 + i),
        )


fn test_dataset_info_cifar10() raises:
    """Test DatasetInfo with CIFAR-10."""
    var info = DatasetInfo("cifar10")

    assert_equal(info.num_classes(), 10, "CIFAR-10 has 10 classes")
    assert_equal(
        info.num_train_samples(), 50000, "CIFAR-10 has 50000 training samples"
    )
    assert_equal(
        info.num_test_samples(), 10000, "CIFAR-10 has 10000 test samples"
    )

    var shape = info.image_shape()
    assert_equal(len(shape), 3, "CIFAR-10 images have 3 dimensions")
    assert_equal(shape[0], 3, "CIFAR-10 has 3 channels (RGB)")
    assert_equal(shape[1], 32, "CIFAR-10 images are 32 pixels tall")
    assert_equal(shape[2], 32, "CIFAR-10 images are 32 pixels wide")

    var classes = info.class_names()
    assert_equal(len(classes), 10, "Class names list has 10 items")
    assert_equal(classes[0], "airplane", "First class is airplane")

    var class_name = info.class_name(0)
    assert_equal(class_name, "airplane", "Class name at index 0")

    var desc = info.description()
    assert_true(len(desc) > 0, "Description should not be empty")


fn test_dataset_info_emnist_balanced() raises:
    """Test DatasetInfo with EMNIST Balanced."""
    var info = DatasetInfo("emnist_balanced")

    assert_equal(info.num_classes(), 47, "EMNIST Balanced has 47 classes")
    assert_equal(
        info.num_train_samples(),
        112589,
        "EMNIST Balanced has ~112589 training samples",
    )
    assert_equal(
        info.num_test_samples(),
        18822,
        "EMNIST Balanced has ~18822 test samples",
    )

    var shape = info.image_shape()
    assert_equal(len(shape), 3, "EMNIST images have 3 dimensions")
    assert_equal(shape[0], 1, "EMNIST is grayscale (1 channel)")
    assert_equal(shape[1], 28, "EMNIST images are 28 pixels tall")
    assert_equal(shape[2], 28, "EMNIST images are 28 pixels wide")

    var classes = info.class_names()
    assert_equal(len(classes), 47, "Class names list has 47 items")

    var class_name = info.class_name(0)
    assert_equal(class_name, "0", "First class is digit 0")


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
    """Run all test_constants tests."""
    print("Running test_constants tests...")

    test_cifar10_class_names()
    print("✓ test_cifar10_class_names")

    test_emnist_balanced_classes()
    print("✓ test_emnist_balanced_classes")

    test_emnist_byclass_classes()
    print("✓ test_emnist_byclass_classes")

    test_emnist_bymerge_classes()
    print("✓ test_emnist_bymerge_classes")

    test_emnist_digits_classes()
    print("✓ test_emnist_digits_classes")

    test_emnist_letters_classes()
    print("✓ test_emnist_letters_classes")

    test_dataset_info_cifar10()
    print("✓ test_dataset_info_cifar10")

    test_dataset_info_emnist_balanced()
    print("✓ test_dataset_info_emnist_balanced")

    test_dataset_info_emnist_byclass()
    print("✓ test_dataset_info_emnist_byclass")

    test_dataset_info_emnist_bymerge()
    print("✓ test_dataset_info_emnist_bymerge")

    test_dataset_info_emnist_digits()
    print("✓ test_dataset_info_emnist_digits")

    test_dataset_info_emnist_letters()
    print("✓ test_dataset_info_emnist_letters")

    test_dataset_info_invalid_dataset()
    print("✓ test_dataset_info_invalid_dataset")

    test_dataset_info_class_name_out_of_range()
    print("✓ test_dataset_info_class_name_out_of_range")

    test_dataset_info_image_shape_all_datasets()
    print("✓ test_dataset_info_image_shape_all_datasets")

    test_class_names_not_empty()
    print("✓ test_class_names_not_empty")

    print("\nAll test_constants tests passed!")
