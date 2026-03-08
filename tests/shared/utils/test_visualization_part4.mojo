"""Tests for visualization utilities module - Part 4.

# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_visualization.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md

This module tests gradient flow visualization, detection, and
data image visualization functionality.
"""

from tests.shared.conftest import (
    assert_true,
    assert_false,
    assert_equal,
    assert_not_equal,
)
from shared.utils.visualization import (
    visualize_gradient_flow,
    detect_gradient_issues,
    show_images,
)


# ============================================================================
# Test Gradient Flow Visualization
# ============================================================================


fn test_visualize_gradient_magnitudes() raises:
    """Test visualizing gradient magnitudes by layer."""
    var gradients = List[Float32]()
    gradients.append(0.01)
    gradients.append(0.005)
    gradients.append(0.001)
    gradients.append(0.0005)

    var layer_names = List[String]()
    layer_names.append("conv1")
    layer_names.append("conv2")
    layer_names.append("fc1")
    layer_names.append("fc2")

    var result = visualize_gradient_flow(gradients, layer_names)
    assert_true(result)


fn test_detect_vanishing_gradients() raises:
    """Test detecting vanishing gradient problem."""
    var gradients = List[Float32]()
    gradients.append(0.01)
    gradients.append(0.0001)
    gradients.append(1e-8)  # Very small - vanishing

    var issues = detect_gradient_issues(gradients)
    var has_vanishing = issues[0]
    var has_exploding = issues[1]

    assert_true(has_vanishing)
    assert_false(has_exploding)


fn test_detect_exploding_gradients() raises:
    """Test detecting exploding gradient problem."""
    var gradients = List[Float32]()
    gradients.append(0.01)
    gradients.append(10.0)
    gradients.append(1000.0)  # Very large - exploding

    var issues = detect_gradient_issues(gradients)
    var has_vanishing = issues[0]
    var has_exploding = issues[1]

    assert_false(has_vanishing)
    assert_true(has_exploding)


fn test_detect_both_gradient_issues() raises:
    """Test detecting both vanishing and exploding gradients."""
    var gradients = List[Float32]()
    gradients.append(1e-10)  # Vanishing
    gradients.append(0.01)  # Normal
    gradients.append(1000.0)  # Exploding

    var issues = detect_gradient_issues(gradients)
    var has_vanishing = issues[0]
    var has_exploding = issues[1]

    assert_true(has_vanishing)
    assert_true(has_exploding)


fn test_no_gradient_issues() raises:
    """Test normal gradients without issues."""
    var gradients = List[Float32]()
    gradients.append(0.01)
    gradients.append(0.005)
    gradients.append(0.001)

    var issues = detect_gradient_issues(gradients)
    var has_vanishing = issues[0]
    var has_exploding = issues[1]

    assert_false(has_vanishing)
    assert_false(has_exploding)


fn test_plot_gradient_flow() raises:
    """Test plotting gradient flow through network."""
    var gradients = List[Float32]()
    gradients.append(0.01)
    gradients.append(0.008)
    gradients.append(0.006)

    var layer_names = List[String]()
    layer_names.append("layer1")
    layer_names.append("layer2")
    layer_names.append("layer3")

    var result = visualize_gradient_flow(gradients, layer_names)
    assert_true(result)


fn test_empty_gradients() raises:
    """Test gradient detection with empty list."""
    var gradients = List[Float32]()

    var issues = detect_gradient_issues(gradients)
    var has_vanishing = issues[0]
    var has_exploding = issues[1]

    assert_false(has_vanishing)
    assert_false(has_exploding)


# ============================================================================
# Test Data Visualization
# ============================================================================


fn test_visualize_image_batch() raises:
    """Test visualizing batch of images in grid."""
    var images = List[String]()
    for i in range(16):
        images.append("image_" + String(i) + ".png")

    var labels = List[String]()

    var result = show_images(images, labels, nrow=4)
    assert_true(result)


fn main() raises:
    """Run all tests."""
    print("Test Visualization Utilities - Part 4")
    print("=" * 50)

    print("  test_visualize_gradient_magnitudes...", end="")
    test_visualize_gradient_magnitudes()
    print(" OK")

    print("  test_detect_vanishing_gradients...", end="")
    test_detect_vanishing_gradients()
    print(" OK")

    print("  test_detect_exploding_gradients...", end="")
    test_detect_exploding_gradients()
    print(" OK")

    print("  test_detect_both_gradient_issues...", end="")
    test_detect_both_gradient_issues()
    print(" OK")

    print("  test_no_gradient_issues...", end="")
    test_no_gradient_issues()
    print(" OK")

    print("  test_plot_gradient_flow...", end="")
    test_plot_gradient_flow()
    print(" OK")

    print("  test_empty_gradients...", end="")
    test_empty_gradients()
    print(" OK")

    print("  test_visualize_image_batch...", end="")
    test_visualize_image_batch()
    print(" OK")

    print()
    print("All visualization part 4 tests passed (8/8)")
