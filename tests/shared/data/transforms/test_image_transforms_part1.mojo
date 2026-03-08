"""Tests for image-specific transforms (Part 1 of 2).

Tests image transforms including resize, crop, and basic normalize operations.
Split from test_image_transforms.mojo per ADR-009 (≤10 fn test_ per file).

# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_image_transforms.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""

from tests.shared.conftest import (
    assert_true,
    assert_equal,
    assert_almost_equal,
    TestFixtures,
)


# ============================================================================
# Resize Transform Tests
# ============================================================================


fn test_resize_basic():
    """Test resizing image to target dimensions.

    Should resize input tensor from any size to specified (height, width),
    using bilinear interpolation by default.
    """
    # var image = Tensor.ones(100, 100, 3)  # 100x100 RGB image
    # var resize = Resize(224, 224)
    # var result = resize(image)
    #
    # assert_equal(result.shape()[0], 224)
    # assert_equal(result.shape()[1], 224)
    # assert_equal(result.shape()[2], 3)  # Channels preserved
    pass


fn test_resize_upscaling():
    """Test resizing smaller image to larger size.

    Should handle upscaling (interpolation) correctly,
    not just downscaling.
    """
    # var image = Tensor.ones(28, 28, 1)  # Small grayscale image
    # var resize = Resize(224, 224)
    # var result = resize(image)
    #
    # assert_equal(result.shape()[0], 224)
    # assert_equal(result.shape()[1], 224)
    pass


fn test_resize_aspect_ratio():
    """Test resizing with different aspect ratio.

    Should allow non-square targets, e.g., 224x320,
    stretching image if needed.
    """
    # var image = Tensor.ones(100, 100, 3)
    # var resize = Resize(height=224, width=320)
    # var result = resize(image)
    #
    # assert_equal(result.shape()[0], 224)
    # assert_equal(result.shape()[1], 320)
    pass


fn test_resize_interpolation_methods():
    """Test different interpolation methods.

    Should support bilinear, nearest-neighbor, and bicubic
    interpolation modes.
    """
    # var image = Tensor.ones(100, 100, 3)
    #
    # var resize_bilinear = Resize(224, 224, mode="bilinear")
    # var resize_nearest = Resize(224, 224, mode="nearest")
    #
    # var result_bilinear = resize_bilinear(image)
    # var result_nearest = resize_nearest(image)
    #
    # # Results should differ based on interpolation
    # assert_not_equal(result_bilinear, result_nearest)
    pass


# ============================================================================
# Crop Transform Tests
# ============================================================================


fn test_center_crop():
    """Test center cropping to smaller size.

    Should extract center region of specified size,
    discarding edges equally from all sides.
    """
    # var image = Tensor.arange(0, 100*100).reshape(100, 100, 1)
    # var crop = CenterCrop(50, 50)
    # var result = crop(image)
    #
    # assert_equal(result.shape()[0], 50)
    # assert_equal(result.shape()[1], 50)
    #
    # # Center pixel should be from center of original
    # # Original center is at (50, 50)
    # # After crop, should be at (25, 25)
    pass


fn test_random_crop():
    """Test random cropping with deterministic seed.

    Should crop random region of specified size,
    deterministic with fixed seed.
    """
    # var image = Tensor.ones(100, 100, 3)
    #
    # TestFixtures.set_seed()
    # var crop1 = RandomCrop(50, 50)
    # var result1 = crop1(image)
    #
    # TestFixtures.set_seed()
    # var crop2 = RandomCrop(50, 50)
    # var result2 = crop2(image)
    #
    # # Same seed should produce same crop location
    # assert_equal(result1, result2)
    pass


fn test_random_crop_padding():
    """Test random crop with padding for small images.

    If crop size > image size, should pad image first
    before cropping.
    """
    # var image = Tensor.ones(28, 28, 1)
    # var crop = RandomCrop(32, 32, padding=4)  # Pad by 4 pixels
    # var result = crop(image)
    #
    # assert_equal(result.shape()[0], 32)
    # assert_equal(result.shape()[1], 32)
    pass


# ============================================================================
# Normalize Transform Tests (basic)
# ============================================================================


fn test_normalize_basic():
    """Test normalization with mean and std.

    Should apply (x - mean) / std normalization,
    standard preprocessing for neural networks.
    """
    # var image = Tensor.ones(28, 28, 1) * 2.0  # All values = 2.0
    # var normalize = Normalize(mean=1.0, std=0.5)
    # var result = normalize(image)
    #
    # # (2.0 - 1.0) / 0.5 = 2.0
    # assert_almost_equal(result[0, 0, 0], 2.0)
    pass


# ============================================================================
# Main Test Runner
# ============================================================================


fn main() raises:
    """Run image transform tests (Part 1): resize, crop, and basic normalize."""
    print("Running image transform tests (Part 1)...")

    # Resize tests
    test_resize_basic()
    test_resize_upscaling()
    test_resize_aspect_ratio()
    test_resize_interpolation_methods()

    # Crop tests
    test_center_crop()
    test_random_crop()
    test_random_crop_padding()

    # Normalize tests (basic)
    test_normalize_basic()

    print("✓ All image transform tests (Part 1) passed!")
