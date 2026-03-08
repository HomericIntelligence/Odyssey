# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_results_printer.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""Tests for results printer module - Part 3.

Covers:
- Confusion matrix printer (3 tests)
- Training summary printer (4 tests)
- Integration tests (1 test)
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
# Confusion Matrix Printer Tests (#2353) - continued
# ============================================================================


fn test_print_confusion_matrix_binary() raises:
    """Test confusion matrix for binary classification."""
    print("Testing print_confusion_matrix binary...")

    # Create 2x2 confusion matrix (binary classification)
    var shape: List[Int] = [2, 2]
    var matrix = ExTensor(shape, DType.int32)

    matrix._data.bitcast[Int32]()[0] = 950  # TP
    matrix._data.bitcast[Int32]()[1] = 50  # FP
    matrix._data.bitcast[Int32]()[2] = 30  # FN
    matrix._data.bitcast[Int32]()[3] = 970  # TN

    var class_names = List[String]()
    class_names.append("Negative")
    class_names.append("Positive")

    print_confusion_matrix(matrix, class_names)

    print("   Confusion matrix binary test passed")


fn test_print_confusion_matrix_normalized() raises:
    """Test confusion matrix with normalized values."""
    print("Testing print_confusion_matrix normalized...")

    # Create 2x2 normalized confusion matrix (as percentages)
    var shape: List[Int] = [2, 2]
    var matrix = ExTensor(shape, DType.float32)

    matrix._data.bitcast[Float32]()[0] = 0.95  # 95%
    matrix._data.bitcast[Float32]()[1] = 0.05  # 5%
    matrix._data.bitcast[Float32]()[2] = 0.03  # 3%
    matrix._data.bitcast[Float32]()[3] = 0.97  # 97%

    print_confusion_matrix(matrix)

    print("   Confusion matrix normalized test passed")


fn test_print_confusion_matrix_large() raises:
    """Test confusion matrix with larger number of classes."""
    print("Testing print_confusion_matrix large...")

    # Create 10x10 confusion matrix
    var shape: List[Int] = [10, 10]
    var matrix = ExTensor(shape, DType.int32)

    # Fill with simple pattern
    for i in range(10):
        for j in range(10):
            var idx = i * 10 + j
            if i == j:
                matrix._data.bitcast[Int32]()[idx] = 90
            else:
                matrix._data.bitcast[Int32]()[idx] = 1

    print_confusion_matrix(matrix)

    print("   Confusion matrix large test passed")


# ============================================================================
# Training Summary Printer Tests (#2353)
# ============================================================================


fn test_print_training_summary_basic() raises:
    """Test basic training summary output."""
    print("Testing print_training_summary basic...")

    print_training_summary(
        total_epochs=100,
        best_train_loss=0.034,
        best_test_loss=0.156,
        best_accuracy=0.965,
        best_epoch=87,
    )

    print("   Training summary basic test passed")


fn test_print_training_summary_perfect_training() raises:
    """Test training summary with perfect metrics."""
    print("Testing print_training_summary perfect...")

    print_training_summary(
        total_epochs=50,
        best_train_loss=0.0,
        best_test_loss=0.0,
        best_accuracy=1.0,
        best_epoch=50,
    )

    print("   Training summary perfect test passed")


fn test_print_training_summary_early_stopping() raises:
    """Test training summary with early stopping."""
    print("Testing print_training_summary early stopping...")

    print_training_summary(
        total_epochs=200,
        best_train_loss=0.054,
        best_test_loss=0.203,
        best_accuracy=0.923,
        best_epoch=45,
    )

    print("   Training summary early stopping test passed")


fn test_print_training_summary_large_epochs() raises:
    """Test training summary with many epochs."""
    print("Testing print_training_summary large epochs...")

    print_training_summary(
        total_epochs=1000,
        best_train_loss=0.012,
        best_test_loss=0.089,
        best_accuracy=0.978,
        best_epoch=789,
    )

    print("   Training summary large epochs test passed")


# ============================================================================
# Integration Tests (#2353)
# ============================================================================


fn test_full_training_workflow_output() raises:
    """Test complete training workflow output."""
    print("Testing full training workflow output...")

    print("\n" + "=" * 60)
    print("COMPLETE TRAINING WORKFLOW TEST")
    print("=" * 60 + "\n")

    # Simulate training progression
    for epoch in range(1, 4):
        # Print progress for a few batches
        for batch in range(1, 11, 5):
            print_training_progress(
                epoch=epoch,
                total_epochs=3,
                batch=batch,
                total_batches=10,
                loss=Float32(2.5 / Float64(epoch) - Float64(batch) * 0.05),
                learning_rate=0.01,
            )

        # Print evaluation results
        var train_loss = Float32(2.0 / Float32(epoch))
        var test_loss = Float32(2.2 / Float32(epoch))
        var train_acc = Float32(0.5 + 0.15 * Float32(epoch))
        var test_acc = Float32(0.45 + 0.12 * Float32(epoch))

        print_evaluation_summary(
            epoch=epoch,
            total_epochs=3,
            train_loss=train_loss,
            train_accuracy=train_acc,
            test_loss=test_loss,
            test_accuracy=test_acc,
        )

    # Print per-class accuracy
    var per_class_shape: List[Int] = [5]
    var per_class = ExTensor(per_class_shape, DType.float64)
    per_class._data.bitcast[Float64]()[0] = 0.85
    per_class._data.bitcast[Float64]()[1] = 0.92
    per_class._data.bitcast[Float64]()[2] = 0.88
    per_class._data.bitcast[Float64]()[3] = 0.95
    per_class._data.bitcast[Float64]()[4] = 0.80

    var class_names = List[String]()
    class_names.append("Class0")
    class_names.append("Class1")
    class_names.append("Class2")
    class_names.append("Class3")
    class_names.append("Class4")

    print_per_class_accuracy(per_class, class_names)

    # Print confusion matrix
    var cm_shape: List[Int] = [5, 5]
    var cm = ExTensor(cm_shape, DType.int32)
    for i in range(5):
        for j in range(5):
            var idx = i * 5 + j
            if i == j:
                cm._data.bitcast[Int32]()[idx] = 85
            else:
                cm._data.bitcast[Int32]()[idx] = 3

    print_confusion_matrix(cm, class_names)

    # Print training summary
    print_training_summary(
        total_epochs=3,
        best_train_loss=0.667,
        best_test_loss=0.733,
        best_accuracy=0.77,
        best_epoch=3,
    )

    print("   Full training workflow test passed")


# ============================================================================
# Main Test Runner
# ============================================================================


fn main() raises:
    """Run results printer tests - part 3."""
    print("\n" + "=" * 60)
    print("Running Results Printer Tests - Part 3")
    print("=" * 60 + "\n")

    # Confusion matrix tests - continued
    test_print_confusion_matrix_binary()
    test_print_confusion_matrix_normalized()
    test_print_confusion_matrix_large()

    # Training summary tests
    test_print_training_summary_basic()
    test_print_training_summary_perfect_training()
    test_print_training_summary_early_stopping()
    test_print_training_summary_large_epochs()

    # Integration tests
    test_full_training_workflow_output()

    print("\n" + "=" * 60)
    print("All results printer tests - part 3 passed!")
    print("=" * 60 + "\n")
