# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_results_printer.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""Tests for results printer module - Part 2.

Covers:
- Evaluation summary printer (1 test)
- Per-class accuracy printer (5 tests)
- Confusion matrix printer (2 tests)
"""

from tests.shared.conftest import (
    assert_true,
    assert_false,
    assert_equal,
)
from shared.core import ExTensor, zeros, ones, full
from shared.training.metrics import (
    print_evaluation_summary,
    print_per_class_accuracy,
    print_confusion_matrix,
    print_training_progress,
    print_training_summary,
)
from collections import List


# ============================================================================
# Evaluation Summary Printer Tests (#2353) - continued
# ============================================================================


fn test_print_evaluation_summary_last_epoch() raises:
    """Test evaluation summary on final epoch."""
    print("Testing print_evaluation_summary last epoch...")

    print_evaluation_summary(
        epoch=200,
        total_epochs=200,
        train_loss=0.034,
        train_accuracy=0.965,
        test_loss=0.156,
        test_accuracy=0.952,
    )

    print("   Evaluation summary last epoch test passed")


# ============================================================================
# Per-Class Accuracy Printer Tests (#2353)
# ============================================================================


fn test_print_per_class_accuracy_basic() raises:
    """Test basic per-class accuracy output."""
    print("Testing print_per_class_accuracy basic...")

    # Create simple per-class accuracy tensor
    var shape: List[Int] = [10]
    var accuracies = full(shape, 0.85, DType.float64)

    print_per_class_accuracy(accuracies)

    print("   Per-class accuracy basic test passed")


fn test_print_per_class_accuracy_with_class_names() raises:
    """Test per-class accuracy with class names."""
    print("Testing print_per_class_accuracy with class names...")

    # Create per-class accuracy tensor
    var shape: List[Int] = [3]
    var accuracies = ExTensor(shape, DType.float64)
    accuracies._data.bitcast[Float64]()[0] = 0.92
    accuracies._data.bitcast[Float64]()[1] = 0.88
    accuracies._data.bitcast[Float64]()[2] = 0.95

    # Create class names
    var class_names = List[String]()
    class_names.append("Cat")
    class_names.append("Dog")
    class_names.append("Bird")

    print_per_class_accuracy(accuracies, class_names)

    print("   Per-class accuracy with class names test passed")


fn test_print_per_class_accuracy_varied_values() raises:
    """Test per-class accuracy with varied accuracy values."""
    print("Testing print_per_class_accuracy varied values...")

    # Create varied per-class accuracy tensor
    var shape: List[Int] = [5]
    var accuracies = ExTensor(shape, DType.float64)
    accuracies._data.bitcast[Float64]()[0] = 0.50
    accuracies._data.bitcast[Float64]()[1] = 0.75
    accuracies._data.bitcast[Float64]()[2] = 0.99
    accuracies._data.bitcast[Float64]()[3] = 0.10
    accuracies._data.bitcast[Float64]()[4] = 1.0

    print_per_class_accuracy(accuracies)

    print("   Per-class accuracy varied values test passed")


fn test_print_per_class_accuracy_single_class() raises:
    """Test per-class accuracy with single class."""
    print("Testing print_per_class_accuracy single class...")

    var shape: List[Int] = [1]
    var accuracies = full(shape, 0.95, DType.float64)

    print_per_class_accuracy(accuracies)

    print("   Per-class accuracy single class test passed")


fn test_print_per_class_accuracy_many_classes() raises:
    """Test per-class accuracy with many classes."""
    print("Testing print_per_class_accuracy many classes...")

    var shape: List[Int] = [100]
    var accuracies = full(shape, 0.87, DType.float64)

    print_per_class_accuracy(accuracies)

    print("   Per-class accuracy many classes test passed")


# ============================================================================
# Confusion Matrix Printer Tests (#2353)
# ============================================================================


fn test_print_confusion_matrix_basic() raises:
    """Test basic confusion matrix output."""
    print("Testing print_confusion_matrix basic...")

    # Create simple 3x3 confusion matrix
    var shape: List[Int] = [3, 3]
    var matrix = ExTensor(shape, DType.int32)

    # Initialize with simple pattern (diagonal dominance)
    for i in range(3):
        for j in range(3):
            var idx = i * 3 + j
            if i == j:
                matrix._data.bitcast[Int32]()[idx] = 90
            else:
                matrix._data.bitcast[Int32]()[idx] = 5

    print_confusion_matrix(matrix)

    print("   Confusion matrix basic test passed")


fn test_print_confusion_matrix_with_class_names() raises:
    """Test confusion matrix with class names."""
    print("Testing print_confusion_matrix with class names...")

    # Create 3x3 confusion matrix
    var shape: List[Int] = [3, 3]
    var matrix = ExTensor(shape, DType.int32)

    # Fill with test data
    matrix._data.bitcast[Int32]()[0] = 90  # 0,0
    matrix._data.bitcast[Int32]()[1] = 5  # 0,1
    matrix._data.bitcast[Int32]()[2] = 5  # 0,2
    matrix._data.bitcast[Int32]()[3] = 3  # 1,0
    matrix._data.bitcast[Int32]()[4] = 92  # 1,1
    matrix._data.bitcast[Int32]()[5] = 5  # 1,2
    matrix._data.bitcast[Int32]()[6] = 2  # 2,0
    matrix._data.bitcast[Int32]()[7] = 4  # 2,1
    matrix._data.bitcast[Int32]()[8] = 94  # 2,2

    var class_names = List[String]()
    class_names.append("Cat")
    class_names.append("Dog")
    class_names.append("Bird")

    print_confusion_matrix(matrix, class_names)

    print("   Confusion matrix with class names test passed")


# ============================================================================
# Main Test Runner
# ============================================================================


fn main() raises:
    """Run results printer tests - part 2."""
    print("\n" + "=" * 60)
    print("Running Results Printer Tests - Part 2")
    print("=" * 60 + "\n")

    # Evaluation summary tests - continued
    test_print_evaluation_summary_last_epoch()

    # Per-class accuracy tests
    test_print_per_class_accuracy_basic()
    test_print_per_class_accuracy_with_class_names()
    test_print_per_class_accuracy_varied_values()
    test_print_per_class_accuracy_single_class()
    test_print_per_class_accuracy_many_classes()

    # Confusion matrix tests
    test_print_confusion_matrix_basic()
    test_print_confusion_matrix_with_class_names()

    print("\n" + "=" * 60)
    print("All results printer tests - part 2 passed!")
    print("=" * 60 + "\n")
