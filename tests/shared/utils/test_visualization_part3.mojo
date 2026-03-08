"""Tests for visualization utilities module - Part 3.

# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_visualization.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md

This module tests confusion matrix normalization, metrics, and model
architecture visualization functionality.
"""

from tests.shared.conftest import (
    assert_true,
    assert_false,
    assert_equal,
    assert_not_equal,
)
from shared.utils.visualization import (
    normalize_confusion_matrix,
    compute_matrix_metrics,
    compute_confusion_matrix,
    visualize_model_architecture,
    visualize_tensor_shapes,
)


# ============================================================================
# Test Confusion Matrix (continued)
# ============================================================================


fn test_confusion_matrix_normalization() raises:
    """Test normalizing confusion matrix by row (true labels)."""
    var matrix = List[List[Int]]()
    var row0 = List[Int]()
    row0.append(8)
    row0.append(2)
    matrix.append(row0^)

    var row1 = List[Int]()
    row1.append(1)
    row1.append(9)
    matrix.append(row1^)

    var normalized = normalize_confusion_matrix(matrix)

    # Row 0: [8, 2] -> [0.8, 0.2]
    assert_true(normalized[0][0] > 0.79 and normalized[0][0] < 0.81)
    assert_true(normalized[0][1] > 0.19 and normalized[0][1] < 0.21)

    # Row 1: [1, 9] -> [0.1, 0.9]
    assert_true(normalized[1][0] > 0.09 and normalized[1][0] < 0.11)
    assert_true(normalized[1][1] > 0.89 and normalized[1][1] < 0.91)


fn test_confusion_matrix_accuracy() raises:
    """Test computing accuracy from confusion matrix."""
    var matrix = List[List[Int]]()

    var row0 = List[Int]()
    row0.append(8)
    row0.append(1)
    row0.append(0)
    matrix.append(row0^)

    var row1 = List[Int]()
    row1.append(1)
    row1.append(7)
    row1.append(2)
    matrix.append(row1^)

    var row2 = List[Int]()
    row2.append(0)
    row2.append(1)
    row2.append(9)
    matrix.append(row2^)

    var metrics = compute_matrix_metrics(matrix)
    var accuracy = metrics[0]

    # Accuracy: (8+7+9) / 29 = 0.8275...
    # Total = 8+1+0+1+7+2+0+1+9 = 29, correct = 8+7+9 = 24
    assert_true(accuracy > 0.82 and accuracy < 0.84)


fn test_single_class_confusion_matrix() raises:
    """Test confusion matrix with single class."""
    var y_true = List[Int]()
    y_true.append(0)
    y_true.append(0)
    y_true.append(0)

    var y_pred = List[Int]()
    y_pred.append(0)
    y_pred.append(0)
    y_pred.append(0)

    var matrix = compute_confusion_matrix(y_true, y_pred)
    assert_equal(len(matrix), 1)
    assert_equal(matrix[0][0], 3)


# ============================================================================
# Test Model Architecture Visualization
# ============================================================================


fn test_visualize_simple_model() raises:
    """Test visualizing simple neural network architecture."""
    var layers = List[String]()
    layers.append("Input: (batch, 784)")
    layers.append("Linear: (batch, 128)")
    layers.append("ReLU: (batch, 128)")
    layers.append("Linear: (batch, 10)")

    var result = visualize_model_architecture("SimpleNN", layers)
    assert_true(result)


fn test_visualize_conv_model() raises:
    """Test visualizing convolutional neural network."""
    var layers = List[String]()
    layers.append("Input: (batch, 1, 28, 28)")
    layers.append("Conv2D: (batch, 32, 26, 26)")
    layers.append("ReLU: (batch, 32, 26, 26)")
    layers.append("MaxPool2D: (batch, 32, 13, 13)")
    layers.append("Flatten: (batch, 5408)")
    layers.append("Linear: (batch, 10)")

    var result = visualize_model_architecture("LeNet", layers)
    assert_true(result)


fn test_visualize_model_with_shapes() raises:
    """Test visualizing model with tensor shapes at each layer."""
    var input_shape = List[Int]()
    input_shape.append(1)
    input_shape.append(28)
    input_shape.append(28)
    input_shape.append(1)

    var layer_shapes = List[List[Int]]()

    var shape1 = List[Int]()
    shape1.append(1)
    shape1.append(24)
    shape1.append(24)
    shape1.append(32)
    layer_shapes.append(shape1^)

    var shape2 = List[Int]()
    shape2.append(1)
    shape2.append(12)
    shape2.append(12)
    shape2.append(32)
    layer_shapes.append(shape2^)

    var result = visualize_tensor_shapes(input_shape, layer_shapes)
    assert_true(result)


fn test_save_architecture_diagram() raises:
    """Test saving architecture diagram to file."""
    var layers = List[String]()
    layers.append("Input: (batch, 784)")
    layers.append("Linear: (batch, 10)")

    var result = visualize_model_architecture("Model", layers, "arch.png")
    assert_true(result)


fn main() raises:
    """Run all tests."""
    print("Test Visualization Utilities - Part 3")
    print("=" * 50)

    print("  test_confusion_matrix_normalization...", end="")
    test_confusion_matrix_normalization()
    print(" OK")

    print("  test_confusion_matrix_accuracy...", end="")
    test_confusion_matrix_accuracy()
    print(" OK")

    print("  test_single_class_confusion_matrix...", end="")
    test_single_class_confusion_matrix()
    print(" OK")

    print("  test_visualize_simple_model...", end="")
    test_visualize_simple_model()
    print(" OK")

    print("  test_visualize_conv_model...", end="")
    test_visualize_conv_model()
    print(" OK")

    print("  test_visualize_model_with_shapes...", end="")
    test_visualize_model_with_shapes()
    print(" OK")

    print("  test_save_architecture_diagram...", end="")
    test_save_architecture_diagram()
    print(" OK")

    print()
    print("All visualization part 3 tests passed (7/7)")
