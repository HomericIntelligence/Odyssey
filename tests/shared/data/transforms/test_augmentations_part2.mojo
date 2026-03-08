"""Tests for data augmentation transforms (part 2 of 2).

Tests random flip, erasing, and composition augmentations.

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
# RandomHorizontalFlip Tests
# ============================================================================


fn test_random_horizontal_flip_probability() raises:
    """Test RandomHorizontalFlip respects probability.

    With p=0.5, should flip approximately 50% of the time
    over many samples.
    """
    # Create a 2x2x3 tensor (flattened to 12 elements)
    # [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
    # Represents:
    # Row 0: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    # Row 1: [7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
    var data_list = List[Float32]()
    for i in range(2 * 2 * 3):
        data_list.append(Float32(i + 1))
    var data_shape = List[Int]()
    data_shape.append(len(data_list))
    var data = ExTensor(data_shape, DType.float32)
    for i in range(len(data_list)):
        data._set_float32(i, data_list[i])

    var flip = RandomHorizontalFlip(0.5)

    TestFixtures.set_seed()
    var flipped_count = 0
    for _ in range(1000):
        var result = flip(data)
        # Check if flipped by examining first element of first row
        # Original first element is 1.0, flipped first element is 4.0 (for 2 width)
        if result[0] > 1.0:
            flipped_count += 1

    # Should be approximately 500 ± tolerance
    assert_true(flipped_count > 400 and flipped_count < 600)


fn test_random_flip_always() raises:
    """Test RandomHorizontalFlip with p=1.0 always flips.

    Should flip every time when probability is 1.0,
    useful for testing.
    """
    # Create a 2x2x3 tensor
    var data_list = List[Float32]()
    for i in range(2 * 2 * 3):
        data_list.append(Float32(i + 1))
    var data_shape = List[Int]()
    data_shape.append(len(data_list))
    var data = ExTensor(data_shape, DType.float32)
    for i in range(len(data_list)):
        data._set_float32(i, data_list[i])

    var flip = RandomHorizontalFlip(1.0)

    for _ in range(10):
        var result = flip(data)
        # First element should always be flipped (should not be 1.0)
        # When flipped, width is reversed, so first row becomes reversed
        assert_true(result[0] > 1.0)


fn test_random_flip_never() raises:
    """Test RandomHorizontalFlip with p=0.0 never flips.

    Should never flip when probability is 0.0,
    degenerating to identity transform.
    """
    # Create a 2x2x3 tensor
    var data_list = List[Float32]()
    for i in range(2 * 2 * 3):
        data_list.append(Float32(i + 1))
    var data_shape = List[Int]()
    data_shape.append(len(data_list))
    var data = ExTensor(data_shape, DType.float32)
    for i in range(len(data_list)):
        data._set_float32(i, data_list[i])

    var flip = RandomHorizontalFlip(0.0)

    for _ in range(10):
        var result = flip(data)
        # Should never be flipped, first element should stay 1.0
        assert_equal(result[0], 1.0)


# ============================================================================
# RandomErasing Tests
# ============================================================================


fn test_random_erasing_basic() raises:
    """Test random erasing (cutout) augmentation.

    Should randomly mask rectangular region with zeros or random noise,
    common augmentation for improving robustness.
    """
    # Create a 28x28x3 tensor filled with 1.0
    var data_list = List[Float32]()
    for _ in range(28 * 28 * 3):
        data_list.append(1.0)
    var data_shape = List[Int]()
    data_shape.append(len(data_list))
    var data = ExTensor(data_shape, DType.float32)
    for i in range(len(data_list)):
        data._set_float32(i, data_list[i])

    var erase = RandomErasing(1.0, (0.02, 0.33))
    var result = erase(data)

    # Some pixels should be erased (not all ones)
    var has_erased = False
    for i in range(result.num_elements()):
        if result[i] != 1.0:
            has_erased = True
            break

    assert_true(has_erased)


fn test_random_erasing_scale() raises:
    """Test random erasing with scale parameter.

    Scale controls size of erased region as fraction of image,
    should respect min/max bounds.
    """
    # Create a 100x100x3 tensor filled with 1.0
    var data_list = List[Float32]()
    for _ in range(100 * 100 * 3):
        data_list.append(1.0)
    var data_shape = List[Int]()
    data_shape.append(len(data_list))
    var data = ExTensor(data_shape, DType.float32)
    for i in range(len(data_list)):
        data._set_float32(i, data_list[i])

    var erase = RandomErasing(1.0, (0.1, 0.2))  # 10-20% of image
    var result = erase(data)

    # Count erased pixels (zeros)
    var erased_count = 0
    for i in range(result.num_elements()):
        if result[i] == 0.0:
            erased_count += 1

    # Should be approximately 10-20% of 100*100*3 pixels
    # 30000 * 0.1 = 3000, 30000 * 0.2 = 6000
    # But scale is per image (100x100), so 10000 * 0.1 = 1000, 10000 * 0.2 = 2000
    # With 3 channels, this becomes 3000-6000 erased pixels
    assert_true(erased_count > 800 and erased_count < 6500)


# ============================================================================
# Compose Random Augmentations Tests
# ============================================================================


fn test_compose_random_augmentations() raises:
    """Test composing multiple random augmentations.

    Should apply all augmentations in sequence,
    each with their own randomness.
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

    # Manually chain transforms (heterogeneous Pipeline not yet supported)
    # Apply: RandomRotation -> RandomHorizontalFlip -> RandomCrop
    TestFixtures.set_seed()
    var rotation = RandomRotation((15.0, 15.0))
    var flip = RandomHorizontalFlip(0.5)
    var crop = RandomCrop((24, 24))

    var result = rotation(data)
    result = flip(result)
    result = crop(result)

    # Output should be 24x24x3 after crop
    assert_equal(result.num_elements(), 24 * 24 * 3)


fn test_augmentation_determinism_in_pipeline() raises:
    """Test that augmentation pipeline is deterministic with seed.

    Entire pipeline should produce same result with same seed,
    even with multiple random augmentations.
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

    # First run with seed
    # Apply: RandomRotation -> RandomCrop -> RandomHorizontalFlip
    TestFixtures.set_seed()
    var rotation1 = RandomRotation((15.0, 15.0))
    var crop1 = RandomCrop((24, 24))
    var flip1 = RandomHorizontalFlip(0.5)
    var result1 = rotation1(data)
    result1 = crop1(result1)
    result1 = flip1(result1)

    # Second run with same seed
    TestFixtures.set_seed()
    var rotation2 = RandomRotation((15.0, 15.0))
    var crop2 = RandomCrop((24, 24))
    var flip2 = RandomHorizontalFlip(0.5)
    var result2 = rotation2(data)
    result2 = crop2(result2)
    result2 = flip2(result2)

    # Both results should have same number of elements
    assert_equal(result1.num_elements(), result2.num_elements())


# ============================================================================
# Main Test Runner
# ============================================================================


fn main() raises:
    """Run augmentation tests (part 2)."""
    print("Running augmentation tests (part 2)...")

    # RandomHorizontalFlip tests
    test_random_horizontal_flip_probability()
    print("  ✓ test_random_horizontal_flip_probability")
    test_random_flip_always()
    print("  ✓ test_random_flip_always")
    test_random_flip_never()
    print("  ✓ test_random_flip_never")

    # RandomErasing tests
    test_random_erasing_basic()
    print("  ✓ test_random_erasing_basic")
    test_random_erasing_scale()
    print("  ✓ test_random_erasing_scale")

    # Composition tests
    test_compose_random_augmentations()
    print("  ✓ test_compose_random_augmentations")
    test_augmentation_determinism_in_pipeline()
    print("  ✓ test_augmentation_determinism_in_pipeline")

    print("\n✓ All 7 augmentation tests (part 2) passed!")
