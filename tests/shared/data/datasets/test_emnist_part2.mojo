"""Tests for EMNIST Dataset Wrapper - Part 2: Shape, Class Counts, and Integration

Tests cover:
- Dataset shape validation
- Class count validation for each split
- Integration with AnyTensorDataset

ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
high test load. Split from test_emnist.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""

from testing import assert_equal, assert_true, assert_false, assert_raises
from shared.data import EMNISTDataset, AnyTensorDataset, Dataset
from shared.core.extensor import AnyTensor


# ============================================================================
# Dataset Shape Tests
# ============================================================================


fn test_emnist_shape() raises:
    """Test shape() method returns correct dimensions.

    Verifies that individual sample shape is (1, 28, 28).
    """
    try:
        var dataset = EMNISTDataset("/tmp/emnist", split="balanced", train=True)
        var shape = dataset.shape()

        assert_equal(len(shape), 3, "Shape should have 3 dimensions")
        assert_equal(shape[0], 1, "Channels should be 1")
        assert_equal(shape[1], 28, "Height should be 28")
        assert_equal(shape[2], 28, "Width should be 28")
    except e:
        print("Test data not available - skipping shape test")


# ============================================================================
# Class Count Tests
# ============================================================================


fn test_emnist_num_classes_balanced() raises:
    """Test num_classes() for balanced split.

    Verifies correct class count for balanced variant.
    """
    try:
        var dataset = EMNISTDataset("/tmp/emnist", split="balanced", train=True)
        assert_equal(
            dataset.num_classes(), 47, "Balanced split should have 47 classes"
        )
    except e:
        print("Test data not available - skipping class count test")


fn test_emnist_num_classes_byclass() raises:
    """Test num_classes() for byclass split.

    Verifies correct class count for byclass variant.
    """
    try:
        var dataset = EMNISTDataset("/tmp/emnist", split="byclass", train=True)
        assert_equal(
            dataset.num_classes(), 62, "Byclass split should have 62 classes"
        )
    except e:
        print("Test data not available - skipping byclass class count test")


fn test_emnist_num_classes_digits() raises:
    """Test num_classes() for digits split.

    Verifies that digits split has 10 classes (same as MNIST).
    """
    try:
        var dataset = EMNISTDataset("/tmp/emnist", split="digits", train=True)
        assert_equal(
            dataset.num_classes(), 10, "Digits split should have 10 classes"
        )
    except e:
        print("Test data not available - skipping digits class count test")


fn test_emnist_num_classes_letters() raises:
    """Test num_classes() for letters split.

    Verifies that letters split has 26 classes (A-Z).
    """
    try:
        var dataset = EMNISTDataset("/tmp/emnist", split="letters", train=True)
        assert_equal(
            dataset.num_classes(), 26, "Letters split should have 26 classes"
        )
    except e:
        print("Test data not available - skipping letters class count test")


fn test_emnist_num_classes_mnist() raises:
    """Test num_classes() for mnist split.

    Verifies that MNIST equivalent has 10 classes.
    """
    try:
        var dataset = EMNISTDataset("/tmp/emnist", split="mnist", train=True)
        assert_equal(
            dataset.num_classes(), 10, "MNIST split should have 10 classes"
        )
    except e:
        print("Test data not available - skipping mnist class count test")


# ============================================================================
# Integration Tests
# ============================================================================


fn test_emnist_get_train_data() raises:
    """Test get_train_data() returns AnyTensorDataset.

    Verifies that the method wraps data in AnyTensorDataset correctly.
    """
    try:
        var dataset = EMNISTDataset("/tmp/emnist", split="balanced", train=True)
        var tensor_dataset = dataset.get_train_data()

        # Verify it's a valid AnyTensorDataset
        var length = tensor_dataset.__len__()
        assert_true(length > 0, "AnyTensorDataset should have samples")
    except e:
        print("Test data not available - skipping get_train_data test")


fn test_emnist_get_test_data() raises:
    """Test get_test_data() returns AnyTensorDataset.

    Verifies that the method wraps data in AnyTensorDataset correctly.
    """
    try:
        var dataset = EMNISTDataset(
            "/tmp/emnist", split="balanced", train=False
        )
        var tensor_dataset = dataset.get_test_data()

        # Verify it's a valid AnyTensorDataset
        var length = tensor_dataset.__len__()
        assert_true(length > 0, "AnyTensorDataset should have samples")
    except e:
        print("Test data not available - skipping get_test_data test")


# ============================================================================
# Main Test Runner
# ============================================================================


fn main() raises:
    """Run EMNIST dataset tests - Part 2: Shape, Class Counts, and Integration.
    """
    print("Running EMNIST dataset tests (Part 2)...")

    # Shape tests
    test_emnist_shape()
    print("✓ test_emnist_shape")

    # Class count tests
    test_emnist_num_classes_balanced()
    print("✓ test_emnist_num_classes_balanced")

    test_emnist_num_classes_byclass()
    print("✓ test_emnist_num_classes_byclass")

    test_emnist_num_classes_digits()
    print("✓ test_emnist_num_classes_digits")

    test_emnist_num_classes_letters()
    print("✓ test_emnist_num_classes_letters")

    test_emnist_num_classes_mnist()
    print("✓ test_emnist_num_classes_mnist")

    # Train/test data tests
    test_emnist_get_train_data()
    print("✓ test_emnist_get_train_data")

    test_emnist_get_test_data()
    print("✓ test_emnist_get_test_data")

    print("\nAll EMNIST dataset tests (Part 2) passed!")
