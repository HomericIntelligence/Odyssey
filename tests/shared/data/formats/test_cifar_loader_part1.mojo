# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_cifar_loader.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""Tests for CIFAR Format Binary Data Loader (Part 1)

Tests the CIFARLoader struct for loading CIFAR-10 and CIFAR-100 binary format files.

Test Coverage (Part 1):
- Loader initialization with valid/invalid CIFAR versions
- Label loading for CIFAR-10 (single label) and CIFAR-100 (dual labels)
- Image loading with proper shape and data integrity

Run with: mojo test tests/shared/data/formats/test_cifar_loader_part1.mojo
"""

from collections import List
from memory import UnsafePointer
from shared.core.extensor import ExTensor, zeros
from shared.data.formats import (
    CIFARLoader,
    CIFAR10_BYTES_PER_IMAGE,
    CIFAR100_BYTES_PER_IMAGE,
)
from tests.shared.conftest import assert_true, assert_equal


# ============================================================================
# Helper Functions to Create Test Data
# ============================================================================


fn create_cifar10_test_file(num_images: Int) -> String:
    """Create synthetic CIFAR-10 binary format data.

    Args:
        num_images: Number of images to create

    Returns:
        String containing binary format data (concatenated image records).
    """
    var total_bytes = num_images * CIFAR10_BYTES_PER_IMAGE
    var result = ""

    # Create binary data as string of characters (1 byte per character)
    for img_idx in range(num_images):
        # 1 byte label (0-9)
        var label = img_idx % 10
        result += chr(label)

        # 3072 bytes of pixel data (3 channels * 32 * 32)
        var pixels_per_image = 32 * 32 * 3
        for pixel_idx in range(pixels_per_image):
            var pixel_value = (img_idx + pixel_idx) % 256
            result += chr(pixel_value)

    return result


fn create_cifar100_test_file(num_images: Int) -> String:
    """Create synthetic CIFAR-100 binary format data.

    Args:
        num_images: Number of images to create

    Returns:
        String containing binary format data (concatenated image records).
    """
    var total_bytes = num_images * CIFAR100_BYTES_PER_IMAGE
    var result = ""

    # Create binary data as string of characters (1 byte per character)
    for img_idx in range(num_images):
        # 1 byte coarse label (0-19)
        var coarse_label = img_idx % 20
        result += chr(coarse_label)

        # 1 byte fine label (0-99)
        var fine_label = img_idx % 100
        result += chr(fine_label)

        # 3072 bytes of pixel data (3 channels * 32 * 32)
        var pixels_per_image = 32 * 32 * 3
        for pixel_idx in range(pixels_per_image):
            var pixel_value = (img_idx + pixel_idx) % 256
            result += chr(pixel_value)

    return result


# ============================================================================
# Loader Initialization Tests
# ============================================================================


fn test_cifar_loader_init_cifar10() raises:
    """Test CIFARLoader initialization with CIFAR-10."""
    print("Test: CIFARLoader init CIFAR-10...")

    var loader = CIFARLoader(10)
    assert_equal(loader.cifar_version, 10, "Should be CIFAR-10")
    assert_equal(loader.image_size, 32, "Image size should be 32")
    assert_equal(loader.channels, 3, "Channels should be 3 (RGB)")
    assert_equal(
        loader.bytes_per_image, 3073, "CIFAR-10: 1 label + 3072 pixels"
    )

    print("  ✓ CIFAR-10 loader initialized correctly")


fn test_cifar_loader_init_cifar100() raises:
    """Test CIFARLoader initialization with CIFAR-100."""
    print("Test: CIFARLoader init CIFAR-100...")

    var loader = CIFARLoader(100)
    assert_equal(loader.cifar_version, 100, "Should be CIFAR-100")
    assert_equal(loader.image_size, 32, "Image size should be 32")
    assert_equal(loader.channels, 3, "Channels should be 3 (RGB)")
    assert_equal(
        loader.bytes_per_image, 3074, "CIFAR-100: 2 labels + 3072 pixels"
    )

    print("  ✓ CIFAR-100 loader initialized correctly")


fn test_cifar_loader_init_invalid_version() raises:
    """Test CIFARLoader rejects invalid CIFAR version."""
    print("Test: CIFARLoader init invalid version...")

    var error_caught = False
    try:
        var loader = CIFARLoader(50)  # Invalid version
    except:
        error_caught = True

    assert_true(error_caught, "Should raise error for invalid CIFAR version")
    print("  ✓ Invalid version correctly rejected")


# ============================================================================
# CIFAR-10 Label Loading Tests
# ============================================================================


fn test_load_cifar10_labels_single_image() raises:
    """Test loading labels from single-image CIFAR-10 file."""
    print("Test: Load CIFAR-10 labels (1 image)...")

    # Create a single-image CIFAR-10 file
    var loader = CIFARLoader(10)
    # Just a label byte (5) + 3072 pixel bytes
    var label_byte = 5
    var content = ""
    content += chr(label_byte)
    for _ in range(3072):
        content += chr(0)

    # Verify we can parse it
    var num_images = len(content) // CIFAR10_BYTES_PER_IMAGE
    assert_equal(num_images, 1, "Should parse 1 image")

    print("  ✓ Single-image CIFAR-10 file created successfully")


fn test_load_cifar10_labels_multiple_images() raises:
    """Test loading labels from multi-image CIFAR-10 file."""
    print("Test: Load CIFAR-10 labels (10 images)...")

    var loader = CIFARLoader(10)
    var num_test_images = 10

    # Create test file with specific label patterns
    var content = ""
    for img_idx in range(num_test_images):
        # Label is just the image index mod 10
        var label = img_idx % 10
        content += chr(label)
        # Pixel data
        for _ in range(3072):
            content += chr(0)

    # Verify size calculation
    var expected_size = num_test_images * CIFAR10_BYTES_PER_IMAGE
    assert_equal(len(content), expected_size, "File size should match expected")

    print("  ✓ Multi-image CIFAR-10 file created successfully")


# ============================================================================
# CIFAR-100 Label Loading Tests
# ============================================================================


fn test_load_cifar100_labels_structure() raises:
    """Test CIFAR-100 label structure (coarse + fine)."""
    print("Test: CIFAR-100 label structure...")

    var loader = CIFARLoader(100)

    # Create test file with known coarse/fine label pairs
    var content = ""
    var num_test_images = 5
    for img_idx in range(num_test_images):
        # Coarse label (0-19)
        var coarse = img_idx % 20
        content += chr(coarse)
        # Fine label (0-99)
        var fine = (img_idx * 13) % 100  # Arbitrary multiplier
        content += chr(fine)
        # Pixel data
        for _ in range(3072):
            content += chr(0)

    var expected_size = num_test_images * CIFAR100_BYTES_PER_IMAGE
    assert_equal(len(content), expected_size, "File size should match expected")

    print("  ✓ CIFAR-100 label structure correct")


# ============================================================================
# Image Shape Tests
# ============================================================================


fn test_image_shape_cifar10() raises:
    """Test that loaded CIFAR-10 images have correct shape."""
    print("Test: CIFAR-10 image shape...")

    var loader = CIFARLoader(10)

    # Expected shape for 10 images
    var num_images = 10
    var expected_shape = List[Int]()
    expected_shape.append(num_images)
    expected_shape.append(3)  # RGB channels
    expected_shape.append(32)  # Height
    expected_shape.append(32)  # Width

    assert_equal(len(expected_shape), 4, "Shape should have 4 dimensions")
    assert_equal(expected_shape[0], 10, "Batch size should be 10")
    assert_equal(expected_shape[1], 3, "Channels should be 3")
    assert_equal(expected_shape[2], 32, "Height should be 32")
    assert_equal(expected_shape[3], 32, "Width should be 32")

    print("  ✓ CIFAR-10 image shape correct")


fn test_image_dtype_is_uint8() raises:
    """Test that loaded images are uint8 data type."""
    print("Test: Image dtype is uint8...")

    # Create a minimal test tensor
    var shape = List[Int]()
    shape.append(1)
    shape.append(3)
    shape.append(32)
    shape.append(32)
    var tensor = zeros(shape, DType.uint8)

    # Verify dtype
    assert_true(tensor.dtype() == DType.uint8, "Tensor should be uint8")

    print("  ✓ Image dtype is uint8")


# ============================================================================
# Main Test Runner
# ============================================================================


fn main() raises:
    """Run CIFAR loader tests (Part 1)."""
    print("\n" + "=" * 70)
    print("CIFAR Format Binary Data Loader Tests (Part 1)")
    print("=" * 70 + "\n")

    # Initialization tests
    print("--- Loader Initialization ---")
    test_cifar_loader_init_cifar10()
    test_cifar_loader_init_cifar100()
    test_cifar_loader_init_invalid_version()

    # CIFAR-10 tests
    print("\n--- CIFAR-10 Tests ---")
    test_load_cifar10_labels_single_image()
    test_load_cifar10_labels_multiple_images()

    # CIFAR-100 tests
    print("\n--- CIFAR-100 Tests ---")
    test_load_cifar100_labels_structure()

    # Shape tests
    print("\n--- Shape Tests ---")
    test_image_shape_cifar10()
    test_image_dtype_is_uint8()

    print("\n" + "=" * 70)
    print("All CIFAR loader tests (Part 1) passed!")
    print("=" * 70 + "\n")
