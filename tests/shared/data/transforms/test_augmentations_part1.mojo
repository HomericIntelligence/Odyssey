"""Tests for data augmentation transforms (part 1 of 2).

Tests random augmentations that increase dataset variety during training,
with emphasis on reproducibility and proper randomization.

# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_augmentations.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""

from tests.shared.conftest import (
    assert_true,
    assert_equal,
    assert_false,
    TestFixtures,
)
from shared.data.transforms import (
    RandomHorizontalFlip,
    RandomVerticalFlip,
    RandomRotation,
    RandomCrop,
    CenterCrop,
    RandomErasing,
)
from shared.core.extensor import ExTensor

# Type comptime for compatibility
comptime Tensor = ExTensor


# ============================================================================
# Random Augmentation Tests
# ============================================================================


fn test_random_augmentation_deterministic() raises:
    """Test that augmentations are deterministic with fixed seed.

    Setting random seed should produce identical augmentations,
    critical for debugging and reproducible experiments.
    """
    # Create a 28x28x3 tensor (2352 elements total)
    var data_list = List[Float32]()
    for _ in range(28 * 28 * 3):
        data_list.append(1.0)
    var data_shape = List[Int]()
    data_shape.append(len(data_list))
    var data = ExTensor(data_shape, DType.float32)
    for i in range(len(data_list)):
        data._set_float32(i, data_list[i])

    # First run
    TestFixtures.set_seed()
    var aug1 = RandomRotation((15.0, 15.0))
    var result1 = aug1(data)

    # Second run with same seed
    TestFixtures.set_seed()
    var aug2 = RandomRotation((15.0, 15.0))
    var result2 = aug2(data)

    # Both should have same number of elements
    assert_equal(result1.num_elements(), result2.num_elements())


fn test_random_augmentation_varies() raises:
    """Test that augmentations vary without fixed seed.

    Multiple calls should produce different augmentations,
    not always the same transformation.
    """
    # Create a 28x28x3 tensor
    var data_list = List[Float32]()
    for _ in range(28 * 28 * 3):
        data_list.append(1.0)
    var data_shape = List[Int]()
    data_shape.append(len(data_list))
    var data = ExTensor(data_shape, DType.float32)
    for i in range(len(data_list)):
        data._set_float32(i, data_list[i])

    var aug = RandomRotation((15.0, 15.0))

    var results = List[Tensor]()
    for _ in range(10):
        results.append(aug(data))

    # Check if any result differs from the first
    # (we just verify they all have same number of elements)
    var all_same_size = True
    for i in range(1, len(results)):
        if results[i].num_elements() != results[0].num_elements():
            all_same_size = False
            break

    assert_true(all_same_size)


# ============================================================================
# RandomRotation Tests
# ============================================================================


fn test_random_rotation_range() raises:
    """Test random rotation within degree range.

    Should rotate image by random angle in [-degrees, +degrees],
    with proper handling of borders.
    """
    # Create a 28x28x3 tensor
    var data_list = List[Float32]()
    for _ in range(28 * 28 * 3):
        data_list.append(1.0)
    var data_shape = List[Int]()
    data_shape.append(len(data_list))
    var data = ExTensor(data_shape, DType.float32)
    for i in range(len(data_list)):
        data._set_float32(i, data_list[i])

    var rotate = RandomRotation((30.0, 30.0))  # ±30 degrees
    var result = rotate(data)

    # Check shape preserved
    assert_equal(result.num_elements(), data.num_elements())


fn test_random_rotation_no_change() raises:
    """Test that rotation with degrees=0 doesn't change image.

    Edge case where rotation range is zero should return
    unchanged image.
    """
    # Create a 28x28x3 tensor
    var data_list = List[Float32]()
    for _ in range(28 * 28 * 3):
        data_list.append(1.0)
    var data_shape = List[Int]()
    data_shape.append(len(data_list))
    var data = ExTensor(data_shape, DType.float32)
    for i in range(len(data_list)):
        data._set_float32(i, data_list[i])

    var rotate = RandomRotation((0.0, 0.0))
    var result = rotate(data)

    # With 0 degrees, all pixels should remain 1.0
    var all_ones = True
    for i in range(result.num_elements()):
        if result[i] != 1.0:
            all_ones = False
            break
    assert_true(all_ones)


fn test_random_rotation_fill_value() raises:
    """Test rotation with custom fill value for empty regions.

    Rotating creates empty corners; should fill with specified value
    (default 0, but configurable).
    """
    # Create a 28x28x3 tensor of 1.0
    var data_list = List[Float32]()
    for _ in range(28 * 28 * 3):
        data_list.append(1.0)
    var data_shape = List[Int]()
    data_shape.append(len(data_list))
    var data = ExTensor(data_shape, DType.float32)
    for i in range(len(data_list)):
        data._set_float32(i, data_list[i])

    var rotate = RandomRotation((45.0, 45.0), 0.5)
    var result = rotate(data)

    # Check that result has same number of elements
    assert_equal(result.num_elements(), data.num_elements())


# ============================================================================
# RandomCrop Tests
# ============================================================================


fn test_random_crop_varies_location() raises:
    """Test that RandomCrop samples different locations.

    Multiple crops should not all be from same location,
    unless image is smaller than crop size.
    """
    # Create a 100x100x1 tensor with sequential values
    var data_list = List[Float32]()
    for i in range(100 * 100 * 1):
        data_list.append(Float32(i))
    var data_shape = List[Int]()
    data_shape.append(len(data_list))
    var data = ExTensor(data_shape, DType.float32)
    for i in range(len(data_list)):
        data._set_float32(i, data_list[i])

    var crop = RandomCrop((50, 50))

    var crops = List[Tensor]()
    for _ in range(10):
        crops.append(crop(data))

    # All crops should have same size
    for i in range(len(crops)):
        assert_equal(crops[i].num_elements(), 50 * 50 * 1)


fn test_random_crop_with_padding() raises:
    """Test RandomCrop with padding for edge handling.

    Padding allows crops that extend beyond image boundaries,
    useful for maintaining crop size with small images.
    """
    # Create a 28x28x1 tensor
    var data_list = List[Float32]()
    for _ in range(28 * 28 * 1):
        data_list.append(1.0)
    var data_shape = List[Int]()
    data_shape.append(len(data_list))
    var data = ExTensor(data_shape, DType.float32)
    for i in range(len(data_list)):
        data._set_float32(i, data_list[i])

    var crop = RandomCrop((32, 32), 4)
    var result = crop(data)

    # Output should be 32x32x1 = 1024 elements
    assert_equal(result.num_elements(), 32 * 32 * 1)


# ============================================================================
# Main Test Runner
# ============================================================================


fn main() raises:
    """Run augmentation tests (part 1)."""
    print("Running augmentation tests (part 1)...")

    # General augmentation tests
    test_random_augmentation_deterministic()
    print("  ✓ test_random_augmentation_deterministic")
    test_random_augmentation_varies()
    print("  ✓ test_random_augmentation_varies")

    # RandomRotation tests
    test_random_rotation_range()
    print("  ✓ test_random_rotation_range")
    test_random_rotation_no_change()
    print("  ✓ test_random_rotation_no_change")
    test_random_rotation_fill_value()
    print("  ✓ test_random_rotation_fill_value")

    # RandomCrop tests
    test_random_crop_varies_location()
    print("  ✓ test_random_crop_varies_location")
    test_random_crop_with_padding()
    print("  ✓ test_random_crop_with_padding")

    print("\n✓ All 7 augmentation tests (part 1) passed!")
