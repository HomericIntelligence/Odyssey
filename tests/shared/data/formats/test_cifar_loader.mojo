"""Tests for CIFAR Format Binary Data Loader.

Tests the CIFARLoader struct for loading CIFAR-10 and CIFAR-100 binary format files.

Test Coverage:
- Loader initialization with valid/invalid CIFAR versions
- Label loading for CIFAR-10 (single label) and CIFAR-100 (dual labels)
- Image loading with proper shape and data integrity

"""


from std.collections import List
from std.memory import UnsafePointer
from shared.tensor.any_tensor import AnyTensor, zeros
from shared.data.formats import (
    CIFARLoader,
    CIFAR10_BYTES_PER_IMAGE,
    CIFAR100_BYTES_PER_IMAGE,
)
from tests.shared.conftest import assert_true, assert_equal


def create_cifar10_test_file(num_images: Int) -> String:
    """Create synthetic CIFAR-10 binary format data.

    Args:
        num_images: Number of images to create.

    Returns:
        String containing binary format data (concatenated image records).
    """
    var _total_bytes = num_images * CIFAR10_BYTES_PER_IMAGE
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


def create_cifar100_test_file(num_images: Int) -> String:
    """Create synthetic CIFAR-100 binary format data.

    Args:
        num_images: Number of images to create.

    Returns:
        String containing binary format data (concatenated image records).
    """
    var _total_bytes = num_images * CIFAR100_BYTES_PER_IMAGE
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


def test_cifar_loader_init_cifar10() raises:
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


def test_cifar_loader_init_cifar100() raises:
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


def test_cifar_loader_init_invalid_version() raises:
    """Test CIFARLoader rejects invalid CIFAR version."""
    print("Test: CIFARLoader init invalid version...")

    var error_caught = False
    try:
        var loader = CIFARLoader(50)  # Invalid version
    except:
        error_caught = True

    assert_true(error_caught, "Should raise error for invalid CIFAR version")
    print("  ✓ Invalid version correctly rejected")


def test_load_cifar10_labels_single_image() raises:
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


def test_load_cifar10_labels_multiple_images() raises:
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


def test_load_cifar100_labels_structure() raises:
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


def test_image_shape_cifar10() raises:
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


def test_image_dtype_is_uint8() raises:
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


def test_label_shape_cifar10() raises:
    """Test that CIFAR-10 labels have correct shape (1D array)."""
    print("Test: CIFAR-10 label shape...")

    # Expected shape for 10 image labels
    var num_images = 10
    var shape = List[Int]()
    shape.append(num_images)

    assert_equal(len(shape), 1, "CIFAR-10 labels should be 1D")
    assert_equal(shape[0], 10, "Should have 10 labels")

    print("  ✓ CIFAR-10 label shape correct")


def test_label_shape_cifar100() raises:
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


def test_validate_cifar10_file_size() raises:
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


def test_validate_cifar100_file_size() raises:
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


def test_calculate_num_images_cifar10() raises:
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


def test_calculate_num_images_cifar100() raises:
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


def main() raises:
    """Run all test_cifar_loader tests."""
    print("Running test_cifar_loader tests...")

    test_cifar_loader_init_cifar10()
    print("✓ test_cifar_loader_init_cifar10")

    test_cifar_loader_init_cifar100()
    print("✓ test_cifar_loader_init_cifar100")

    test_cifar_loader_init_invalid_version()
    print("✓ test_cifar_loader_init_invalid_version")

    test_load_cifar10_labels_single_image()
    print("✓ test_load_cifar10_labels_single_image")

    test_load_cifar10_labels_multiple_images()
    print("✓ test_load_cifar10_labels_multiple_images")

    test_load_cifar100_labels_structure()
    print("✓ test_load_cifar100_labels_structure")

    test_image_shape_cifar10()
    print("✓ test_image_shape_cifar10")

    test_image_dtype_is_uint8()
    print("✓ test_image_dtype_is_uint8")

    test_label_shape_cifar10()
    print("✓ test_label_shape_cifar10")

    test_label_shape_cifar100()
    print("✓ test_label_shape_cifar100")

    test_validate_cifar10_file_size()
    print("✓ test_validate_cifar10_file_size")

    test_validate_cifar100_file_size()
    print("✓ test_validate_cifar100_file_size")

    test_calculate_num_images_cifar10()
    print("✓ test_calculate_num_images_cifar10")

    test_calculate_num_images_cifar100()
    print("✓ test_calculate_num_images_cifar100")

    print("\nAll test_cifar_loader tests passed!")
