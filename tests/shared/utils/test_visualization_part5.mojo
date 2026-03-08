"""Tests for visualization utilities module - Part 5.

# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_visualization.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md

This module tests augmented image visualization, feature maps,
and plot export functionality.
"""

from tests.shared.conftest import (
    assert_true,
    assert_false,
    assert_equal,
    assert_not_equal,
)
from shared.utils.visualization import (
    show_images,
    show_augmented_images,
    visualize_feature_maps,
    save_figure,
    clear_figure,
    show_figure,
)


# ============================================================================
# Test Data Visualization (continued)
# ============================================================================


fn test_visualize_images_with_labels() raises:
    """Test visualizing images with labels."""
    var images = List[String]()
    images.append("img1.png")
    images.append("img2.png")
    images.append("img3.png")

    var labels = List[String]()
    labels.append("cat")
    labels.append("dog")
    labels.append("bird")

    var result = show_images(images, labels)
    assert_true(result)


fn test_visualize_augmented_images() raises:
    """Test visualizing original and augmented images side by side."""
    var original = List[String]()
    original.append("orig1.png")
    original.append("orig2.png")

    var augmented = List[String]()
    augmented.append("aug1.png")
    augmented.append("aug2.png")

    var result = show_augmented_images(original, augmented, nrow=2)
    assert_true(result)


# ============================================================================
# Test Feature Map Visualization
# ============================================================================


fn test_visualize_feature_maps() raises:
    """Test visualizing convolutional feature maps."""
    var feature_maps = List[String]()
    feature_maps.append("fmap_0")
    feature_maps.append("fmap_1")
    feature_maps.append("fmap_2")
    feature_maps.append("fmap_3")

    var result = visualize_feature_maps(feature_maps, "conv1")
    assert_true(result)


fn test_visualize_feature_maps_no_layer_name() raises:
    """Test visualizing feature maps without layer name."""
    var feature_maps = List[String]()
    feature_maps.append("fmap_0")

    var result = visualize_feature_maps(feature_maps)
    assert_true(result)


# ============================================================================
# Test Plot Export
# ============================================================================


fn test_save_figure_png() raises:
    """Test saving figure as PNG."""
    var result = save_figure("output.png", "png")
    assert_true(result)


fn test_save_figure_svg() raises:
    """Test saving figure as SVG."""
    var result = save_figure("output.svg", "svg")
    assert_true(result)


fn test_save_figure_pdf() raises:
    """Test saving figure as PDF."""
    var result = save_figure("output.pdf", "pdf")
    assert_true(result)


fn test_clear_figure() raises:
    """Test clearing figure."""
    var result = clear_figure()
    assert_true(result)


fn test_show_figure() raises:
    """Test showing figure."""
    var result = show_figure()
    assert_true(result)


fn main() raises:
    """Run all tests."""
    print("Test Visualization Utilities - Part 5")
    print("=" * 50)

    print("  test_visualize_images_with_labels...", end="")
    test_visualize_images_with_labels()
    print(" OK")

    print("  test_visualize_augmented_images...", end="")
    test_visualize_augmented_images()
    print(" OK")

    print("  test_visualize_feature_maps...", end="")
    test_visualize_feature_maps()
    print(" OK")

    print("  test_visualize_feature_maps_no_layer_name...", end="")
    test_visualize_feature_maps_no_layer_name()
    print(" OK")

    print("  test_save_figure_png...", end="")
    test_save_figure_png()
    print(" OK")

    print("  test_save_figure_svg...", end="")
    test_save_figure_svg()
    print(" OK")

    print("  test_save_figure_pdf...", end="")
    test_save_figure_pdf()
    print(" OK")

    print("  test_clear_figure...", end="")
    test_clear_figure()
    print(" OK")

    print("  test_show_figure...", end="")
    test_show_figure()
    print(" OK")

    print()
    print("All visualization part 5 tests passed (9/9)")
