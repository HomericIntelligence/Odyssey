# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_cifar_loader.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""Tests for CIFAR Format Binary Data Loader (Part 2)

Tests the CIFARLoader struct for loading CIFAR-10 and CIFAR-100 binary format files.

Test Coverage (Part 2):
- Label shape validation for CIFAR-10 and CIFAR-100
- File size validation and error handling
- Image count calculation from file size

Run with: mojo test tests/shared/data/formats/test_cifar_loader_part2.mojo
"""

from collections import List
from shared.data.formats import (
    CIFARLoader,
    CIFAR10_BYTES_PER_IMAGE,
    CIFAR100_BYTES_PER_IMAGE,
)
from tests.shared.conftest import assert_true, assert_equal


# ============================================================================
# Label Shape Tests
# ============================================================================


fn test_label_shape_cifar10() raises:
    """Test that CIFAR-10 labels have correct shape (1D array)."""
    print("Test: CIFAR-10 label shape...")

    # Expected shape for 10 image labels
    var num_images = 10
    var shape = List[Int]()
    shape.append(num_images)

    assert_equal(len(shape), 1, "CIFAR-10 labels should be 1D")
    assert_equal(shape[0], 10, "Should have 10 labels")

    print("  ✓ CIFAR-10 label shape correct")


fn test_label_shape_cifar100() raises:
    """Test that CIFAR-100 labels have correct shape (2D array)."""
    print("Test: CIFAR-100 label shape...")

    # Expected shape for 10 images with (coarse, fine) labels
    var num_images = 10
    var shape = List[Int]()
    shape.append(num_images)
    shape.append(2)  # coarse + fine

    assert_equal(len(shape), 2, "CIFAR-100 labels should be 2D")
    assert_equal(shape[0], 10, "Should have 10 images")
    assert_equal(shape[1], 2, "Should have 2 labels per image")

    print("  ✓ CIFAR-100 label shape correct")


# ============================================================================
# File Validation Tests
# ============================================================================


fn test_validate_cifar10_file_size() raises:
    """Test file size validation for CIFAR-10."""
    print("Test: CIFAR-10 file size validation...")

    var loader = CIFARLoader(10)

    # Valid sizes: multiples of 3073
    var valid_sizes: List[Int] = [3073, 6146, 30730]  # 1, 2, 10 images
    for size_idx in range(len(valid_sizes)):
        var size = valid_sizes[size_idx]
        try:
            loader._validate_file_size(size)
        except e:
            raise Error("Valid size " + String(size) + " was rejected")

    print("  ✓ Valid CIFAR-10 sizes accepted")

    # Invalid size: not a multiple of 3073
    var error_caught = False
    try:
        loader._validate_file_size(1000)
    except:
        error_caught = True

    assert_true(error_caught, "Invalid size should be rejected")
    print("  ✓ Invalid CIFAR-10 sizes rejected")


fn test_validate_cifar100_file_size() raises:
    """Test file size validation for CIFAR-100."""
    print("Test: CIFAR-100 file size validation...")

    var loader = CIFARLoader(100)

    # Valid sizes: multiples of 3074
    var valid_sizes: List[Int] = [3074, 6148, 30740]  # 1, 2, 10 images
    for size_idx in range(len(valid_sizes)):
        var size = valid_sizes[size_idx]
        try:
            loader._validate_file_size(size)
        except e:
            raise Error("Valid size " + String(size) + " was rejected")

    print("  ✓ Valid CIFAR-100 sizes accepted")

    # Invalid size: not a multiple of 3074
    var error_caught = False
    try:
        loader._validate_file_size(1000)
    except:
        error_caught = True

    assert_true(error_caught, "Invalid size should be rejected")
    print("  ✓ Invalid CIFAR-100 sizes rejected")


# ============================================================================
# Image Count Calculation Tests
# ============================================================================


fn test_calculate_num_images_cifar10() raises:
    """Test calculating number of images from file size (CIFAR-10)."""
    print("Test: Calculate num images (CIFAR-10)...")

    var loader = CIFARLoader(10)

    # Test various image counts
    var test_counts: List[Int] = [1, 2, 5, 10, 100]
    for count_idx in range(len(test_counts)):
        var expected_count = test_counts[count_idx]
        var file_size = expected_count * CIFAR10_BYTES_PER_IMAGE
        var calculated_count = loader._calculate_num_images(file_size)
        assert_equal(
            calculated_count,
            expected_count,
            "Should calculate correct image count",
        )

    print("  ✓ Image count calculation correct (CIFAR-10)")


fn test_calculate_num_images_cifar100() raises:
    """Test calculating number of images from file size (CIFAR-100)."""
    print("Test: Calculate num images (CIFAR-100)...")

    var loader = CIFARLoader(100)

    # Test various image counts
    var test_counts: List[Int] = [1, 2, 5, 10, 100]
    for count_idx in range(len(test_counts)):
        var expected_count = test_counts[count_idx]
        var file_size = expected_count * CIFAR100_BYTES_PER_IMAGE
        var calculated_count = loader._calculate_num_images(file_size)
        assert_equal(
            calculated_count,
            expected_count,
            "Should calculate correct image count",
        )

    print("  ✓ Image count calculation correct (CIFAR-100)")


# ============================================================================
# Main Test Runner
# ============================================================================


fn main() raises:
    """Run CIFAR loader tests (Part 2)."""
    print("\n" + "=" * 70)
    print("CIFAR Format Binary Data Loader Tests (Part 2)")
    print("=" * 70 + "\n")

    # Label shape tests
    print("--- Label Shape Tests ---")
    test_label_shape_cifar10()
    test_label_shape_cifar100()

    # Validation tests
    print("\n--- File Validation Tests ---")
    test_validate_cifar10_file_size()
    test_validate_cifar100_file_size()

    # Calculation tests
    print("\n--- Image Count Calculation ---")
    test_calculate_num_images_cifar10()
    test_calculate_num_images_cifar100()

    print("\n" + "=" * 70)
    print("All CIFAR loader tests (Part 2) passed!")
    print("=" * 70 + "\n")
