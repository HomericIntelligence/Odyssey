"""Tests for visualization utilities module - Part 2.

# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_visualization.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md

This module tests plot_loss_only, plot_accuracy_only, and confusion matrix
creation and plotting functionality.
"""

from tests.shared.conftest import (
    assert_true,
    assert_false,
    assert_equal,
    assert_not_equal,
)
from shared.utils.visualization import (
    plot_loss_only,
    plot_accuracy_only,
    compute_confusion_matrix,
    plot_confusion_matrix,
)


# ============================================================================
# Test Training Curve Plotting (continued)
# ============================================================================


fn test_plot_loss_only_single_series() raises:
    """Test plotting single loss series."""
    var losses = List[Float32]()
    losses.append(0.5)
    losses.append(0.4)
    losses.append(0.3)

    var result = plot_loss_only(losses, "Training Loss")
    assert_true(result)


fn test_plot_accuracy_only_single_series() raises:
    """Test plotting single accuracy series."""
    var accuracies = List[Float32]()
    accuracies.append(0.6)
    accuracies.append(0.75)
    accuracies.append(0.85)

    var result = plot_accuracy_only(accuracies, "Validation Accuracy")
    assert_true(result)


fn test_plot_with_save_path() raises:
    """Test plotting with save path specified."""
    var losses = List[Float32]()
    losses.append(0.5)
    losses.append(0.3)

    var result = plot_loss_only(losses, "Loss", "output.png")
    assert_true(result)


# ============================================================================
# Test Confusion Matrix
# ============================================================================


fn test_create_confusion_matrix() raises:
    """Test creating confusion matrix from predictions."""
    var y_true = List[Int]()
    y_true.append(0)
    y_true.append(1)
    y_true.append(2)
    y_true.append(0)
    y_true.append(1)
    y_true.append(2)

    var y_pred = List[Int]()
    y_pred.append(0)
    y_pred.append(2)
    y_pred.append(2)
    y_pred.append(0)
    y_pred.append(1)
    y_pred.append(1)

    var matrix = compute_confusion_matrix(y_true, y_pred)

    # Verify matrix shape: 3x3 (for 3 classes)
    assert_equal(len(matrix), 3)
    assert_equal(len(matrix[0]), 3)

    # Verify diagonal values (correct predictions)
    assert_equal(matrix[0][0], 2)  # Class 0: 2 correct
    assert_equal(matrix[1][1], 1)  # Class 1: 1 correct
    assert_equal(matrix[2][2], 1)  # Class 2: 1 correct


fn test_confusion_matrix_with_num_classes() raises:
    """Test confusion matrix with specified number of classes."""
    var y_true = List[Int]()
    y_true.append(0)
    y_true.append(1)

    var y_pred = List[Int]()
    y_pred.append(0)
    y_pred.append(1)

    # Request 4 classes even though only 2 are present
    var matrix = compute_confusion_matrix(y_true, y_pred, num_classes=4)

    assert_equal(len(matrix), 4)
    assert_equal(len(matrix[0]), 4)


fn test_plot_confusion_matrix() raises:
    """Test plotting confusion matrix as heatmap."""
    var y_true = List[Int]()
    y_true.append(0)
    y_true.append(1)
    y_true.append(0)
    y_true.append(1)

    var y_pred = List[Int]()
    y_pred.append(0)
    y_pred.append(0)
    y_pred.append(0)
    y_pred.append(1)

    var class_names = List[String]()
    class_names.append("cat")
    class_names.append("dog")

    var result = plot_confusion_matrix(y_true, y_pred, class_names)
    assert_true(result)


fn test_confusion_matrix_with_class_names() raises:
    """Test confusion matrix with custom class names."""
    var y_true = List[Int]()
    y_true.append(0)
    y_true.append(1)
    y_true.append(2)

    var y_pred = List[Int]()
    y_pred.append(0)
    y_pred.append(1)
    y_pred.append(2)

    var class_names = List[String]()
    class_names.append("cat")
    class_names.append("dog")
    class_names.append("bird")

    var result = plot_confusion_matrix(y_true, y_pred, class_names)
    assert_true(result)


fn test_empty_confusion_matrix() raises:
    """Test confusion matrix with empty inputs."""
    var y_true = List[Int]()
    var y_pred = List[Int]()

    var matrix = compute_confusion_matrix(y_true, y_pred)
    # Empty inputs should produce empty matrix
    assert_equal(len(matrix), 0)


fn main() raises:
    """Run all tests."""
    print("Test Visualization Utilities - Part 2")
    print("=" * 50)

    print("  test_plot_loss_only_single_series...", end="")
    test_plot_loss_only_single_series()
    print(" OK")

    print("  test_plot_accuracy_only_single_series...", end="")
    test_plot_accuracy_only_single_series()
    print(" OK")

    print("  test_plot_with_save_path...", end="")
    test_plot_with_save_path()
    print(" OK")

    print("  test_create_confusion_matrix...", end="")
    test_create_confusion_matrix()
    print(" OK")

    print("  test_confusion_matrix_with_num_classes...", end="")
    test_confusion_matrix_with_num_classes()
    print(" OK")

    print("  test_plot_confusion_matrix...", end="")
    test_plot_confusion_matrix()
    print(" OK")

    print("  test_confusion_matrix_with_class_names...", end="")
    test_confusion_matrix_with_class_names()
    print(" OK")

    print("  test_empty_confusion_matrix...", end="")
    test_empty_confusion_matrix()
    print(" OK")

    print()
    print("All visualization part 2 tests passed (8/8)")
