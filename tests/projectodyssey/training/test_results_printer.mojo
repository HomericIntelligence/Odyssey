"""Tests for results printer module.

Covers:
- Training progress printer (5 tests)
- Evaluation summary printer (3 tests)
"""


from tests.projectodyssey.conftest import (
    assert_true,
    assert_false,
    assert_equal,
)
from projectodyssey.tensor.any_tensor import AnyTensor
from projectodyssey.tensor.tensor_creation import zeros, ones, full
from projectodyssey.training.metrics import (
    print_evaluation_summary,
    print_per_class_accuracy,
    print_confusion_matrix,
    print_training_progress,
    print_training_summary,
)
from std.collections import List


def test_print_training_progress_basic() raises:
    """Test basic training progress output."""
    print("Testing print_training_progress basic...")

    # Should not raise any errors
    print_training_progress(
        epoch=1,
        total_epochs=100,
        batch=10,
        total_batches=50,
        loss=0.523,
        learning_rate=0.01,
    )

    print("   Training progress basic test passed")


def test_print_training_progress_early_epoch() raises:
    """Test training progress in early epoch."""
    print("Testing print_training_progress early epoch...")

    # First batch of first epoch
    print_training_progress(
        epoch=1,
        total_epochs=200,
        batch=1,
        total_batches=100,
        loss=2.453,
        learning_rate=0.1,
    )

    print("   Training progress early epoch test passed")


def test_print_training_progress_late_epoch() raises:
    """Test training progress in late epoch."""
    print("Testing print_training_progress late epoch...")

    # Last batch of last epoch
    print_training_progress(
        epoch=200,
        total_epochs=200,
        batch=1000,
        total_batches=1000,
        loss=0.012,
        learning_rate=0.0001,
    )

    print("   Training progress late epoch test passed")


def test_print_training_progress_very_small_loss() raises:
    """Test training progress with very small loss values."""
    print("Testing print_training_progress with very small loss...")

    print_training_progress(
        epoch=100,
        total_epochs=200,
        batch=50,
        total_batches=100,
        loss=Float32(1e-6),
        learning_rate=Float32(1e-8),
    )

    print("   Training progress very small loss test passed")


def test_print_training_progress_large_loss() raises:
    """Test training progress with large loss values."""
    print("Testing print_training_progress with large loss...")

    print_training_progress(
        epoch=1,
        total_epochs=100,
        batch=1,
        total_batches=100,
        loss=100.234,
        learning_rate=0.1,
    )

    print("   Training progress large loss test passed")


def test_print_evaluation_summary_basic() raises:
    """Test basic evaluation summary output."""
    print("Testing print_evaluation_summary basic...")

    print_evaluation_summary(
        epoch=1,
        total_epochs=100,
        train_loss=0.523,
        train_accuracy=0.923,
        test_loss=0.645,
        test_accuracy=0.891,
    )

    print("   Evaluation summary basic test passed")


def test_print_evaluation_summary_perfect_accuracy() raises:
    """Test evaluation summary with perfect accuracy."""
    print("Testing print_evaluation_summary perfect accuracy...")

    print_evaluation_summary(
        epoch=50,
        total_epochs=100,
        train_loss=0.001,
        train_accuracy=1.0,
        test_loss=0.005,
        test_accuracy=1.0,
    )

    print("   Evaluation summary perfect accuracy test passed")


def test_print_evaluation_summary_zero_accuracy() raises:
    """Test evaluation summary with zero accuracy."""
    print("Testing print_evaluation_summary zero accuracy...")

    print_evaluation_summary(
        epoch=1,
        total_epochs=100,
        train_loss=4.605,
        train_accuracy=0.0,
        test_loss=4.605,
        test_accuracy=0.0,
    )

    print("   Evaluation summary zero accuracy test passed")


def test_print_evaluation_summary_last_epoch() raises:
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


def test_print_per_class_accuracy_basic() raises:
    """Test basic per-class accuracy output."""
    print("Testing print_per_class_accuracy basic...")

    # Create simple per-class accuracy tensor
    var shape: List[Int] = [10]
    var accuracies = full(shape, 0.85, DType.float64)

    print_per_class_accuracy(accuracies)

    print("   Per-class accuracy basic test passed")


def test_print_per_class_accuracy_with_class_names() raises:
    """Test per-class accuracy with class names."""
    print("Testing print_per_class_accuracy with class names...")

    # Create per-class accuracy tensor
    var shape: List[Int] = [3]
    var accuracies = AnyTensor(shape, DType.float64)
    accuracies.set(0, Float64(0.92))
    accuracies.set(1, Float64(0.88))
    accuracies.set(2, Float64(0.95))

    # Create class names
    var class_names = List[String]()
    class_names.append("Cat")
    class_names.append("Dog")
    class_names.append("Bird")

    print_per_class_accuracy(accuracies, class_names)

    print("   Per-class accuracy with class names test passed")


def test_print_per_class_accuracy_varied_values() raises:
    """Test per-class accuracy with varied accuracy values."""
    print("Testing print_per_class_accuracy varied values...")

    # Create varied per-class accuracy tensor
    var shape: List[Int] = [5]
    var accuracies = AnyTensor(shape, DType.float64)
    accuracies.set(0, Float64(0.50))
    accuracies.set(1, Float64(0.75))
    accuracies.set(2, Float64(0.99))
    accuracies.set(3, Float64(0.10))
    accuracies.set(4, Float64(1.0))

    print_per_class_accuracy(accuracies)

    print("   Per-class accuracy varied values test passed")


def test_print_per_class_accuracy_single_class() raises:
    """Test per-class accuracy with single class."""
    print("Testing print_per_class_accuracy single class...")

    var shape: List[Int] = [1]
    var accuracies = full(shape, 0.95, DType.float64)

    print_per_class_accuracy(accuracies)

    print("   Per-class accuracy single class test passed")


def test_print_per_class_accuracy_many_classes() raises:
    """Test per-class accuracy with many classes."""
    print("Testing print_per_class_accuracy many classes...")

    var shape: List[Int] = [100]
    var accuracies = full(shape, 0.87, DType.float64)

    print_per_class_accuracy(accuracies)

    print("   Per-class accuracy many classes test passed")


def test_print_confusion_matrix_basic() raises:
    """Test basic confusion matrix output."""
    print("Testing print_confusion_matrix basic...")

    # Create simple 3x3 confusion matrix
    var shape: List[Int] = [3, 3]
    var matrix = AnyTensor(shape, DType.int32)

    # Initialize with simple pattern (diagonal dominance)
    for i in range(3):
        for j in range(3):
            var idx = i * 3 + j
            if i == j:
                matrix.set(idx, Int32(90))
            else:
                matrix.set(idx, Int32(5))

    print_confusion_matrix(matrix)

    print("   Confusion matrix basic test passed")


def test_print_confusion_matrix_with_class_names() raises:
    """Test confusion matrix with class names."""
    print("Testing print_confusion_matrix with class names...")

    # Create 3x3 confusion matrix
    var shape: List[Int] = [3, 3]
    var matrix = AnyTensor(shape, DType.int32)

    # Fill with test data
    matrix.set(0, Int32(90))  # 0,0
    matrix.set(1, Int32(5))  # 0,1
    matrix.set(2, Int32(5))  # 0,2
    matrix.set(3, Int32(3))  # 1,0
    matrix.set(4, Int32(92))  # 1,1
    matrix.set(5, Int32(5))  # 1,2
    matrix.set(6, Int32(2))  # 2,0
    matrix.set(7, Int32(4))  # 2,1
    matrix.set(8, Int32(94))  # 2,2

    var class_names = List[String]()
    class_names.append("Cat")
    class_names.append("Dog")
    class_names.append("Bird")

    print_confusion_matrix(matrix, class_names)

    print("   Confusion matrix with class names test passed")


def test_print_confusion_matrix_binary() raises:
    """Test confusion matrix for binary classification."""
    print("Testing print_confusion_matrix binary...")

    # Create 2x2 confusion matrix (binary classification)
    var shape: List[Int] = [2, 2]
    var matrix = AnyTensor(shape, DType.int32)

    matrix.set(0, Int32(950))  # TP
    matrix.set(1, Int32(50))  # FP
    matrix.set(2, Int32(30))  # FN
    matrix.set(3, Int32(970))  # TN

    var class_names = List[String]()
    class_names.append("Negative")
    class_names.append("Positive")

    print_confusion_matrix(matrix, class_names)

    print("   Confusion matrix binary test passed")


def test_print_confusion_matrix_normalized() raises:
    """Test confusion matrix with normalized values."""
    print("Testing print_confusion_matrix normalized...")

    # Create 2x2 normalized confusion matrix (as percentages)
    var shape: List[Int] = [2, 2]
    var matrix = AnyTensor(shape, DType.float32)

    matrix.set(0, Float32(0.95))  # 95%
    matrix.set(1, Float32(0.05))  # 5%
    matrix.set(2, Float32(0.03))  # 3%
    matrix.set(3, Float32(0.97))  # 97%

    print_confusion_matrix(matrix)

    print("   Confusion matrix normalized test passed")


def test_print_confusion_matrix_large() raises:
    """Test confusion matrix with larger number of classes."""
    print("Testing print_confusion_matrix large...")

    # Create 10x10 confusion matrix
    var shape: List[Int] = [10, 10]
    var matrix = AnyTensor(shape, DType.int32)

    # Fill with simple pattern
    for i in range(10):
        for j in range(10):
            var idx = i * 10 + j
            if i == j:
                matrix.set(idx, Int32(90))
            else:
                matrix.set(idx, Int32(1))

    print_confusion_matrix(matrix)

    print("   Confusion matrix large test passed")


def test_print_training_summary_basic() raises:
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


def test_print_training_summary_perfect_training() raises:
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


def test_print_training_summary_early_stopping() raises:
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


def test_print_training_summary_large_epochs() raises:
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


def test_full_training_workflow_output() raises:
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
    var per_class = AnyTensor(per_class_shape, DType.float64)
    per_class.set(0, Float64(0.85))
    per_class.set(1, Float64(0.92))
    per_class.set(2, Float64(0.88))
    per_class.set(3, Float64(0.95))
    per_class.set(4, Float64(0.80))

    var class_names = List[String]()
    class_names.append("Class0")
    class_names.append("Class1")
    class_names.append("Class2")
    class_names.append("Class3")
    class_names.append("Class4")

    print_per_class_accuracy(per_class, class_names)

    # Print confusion matrix
    var cm_shape: List[Int] = [5, 5]
    var cm = AnyTensor(cm_shape, DType.int32)
    for i in range(5):
        for j in range(5):
            var idx = i * 5 + j
            if i == j:
                cm.set(idx, Int32(85))
            else:
                cm.set(idx, Int32(3))

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


def main() raises:
    """Run all test_results_printer tests."""
    print("Running test_results_printer tests...")

    test_print_training_progress_basic()
    print("✓ test_print_training_progress_basic")

    test_print_training_progress_early_epoch()
    print("✓ test_print_training_progress_early_epoch")

    test_print_training_progress_late_epoch()
    print("✓ test_print_training_progress_late_epoch")

    test_print_training_progress_very_small_loss()
    print("✓ test_print_training_progress_very_small_loss")

    test_print_training_progress_large_loss()
    print("✓ test_print_training_progress_large_loss")

    test_print_evaluation_summary_basic()
    print("✓ test_print_evaluation_summary_basic")

    test_print_evaluation_summary_perfect_accuracy()
    print("✓ test_print_evaluation_summary_perfect_accuracy")

    test_print_evaluation_summary_zero_accuracy()
    print("✓ test_print_evaluation_summary_zero_accuracy")

    test_print_evaluation_summary_last_epoch()
    print("✓ test_print_evaluation_summary_last_epoch")

    test_print_per_class_accuracy_basic()
    print("✓ test_print_per_class_accuracy_basic")

    test_print_per_class_accuracy_with_class_names()
    print("✓ test_print_per_class_accuracy_with_class_names")

    test_print_per_class_accuracy_varied_values()
    print("✓ test_print_per_class_accuracy_varied_values")

    test_print_per_class_accuracy_single_class()
    print("✓ test_print_per_class_accuracy_single_class")

    test_print_per_class_accuracy_many_classes()
    print("✓ test_print_per_class_accuracy_many_classes")

    test_print_confusion_matrix_basic()
    print("✓ test_print_confusion_matrix_basic")

    test_print_confusion_matrix_with_class_names()
    print("✓ test_print_confusion_matrix_with_class_names")

    test_print_confusion_matrix_binary()
    print("✓ test_print_confusion_matrix_binary")

    test_print_confusion_matrix_normalized()
    print("✓ test_print_confusion_matrix_normalized")

    test_print_confusion_matrix_large()
    print("✓ test_print_confusion_matrix_large")

    test_print_training_summary_basic()
    print("✓ test_print_training_summary_basic")

    test_print_training_summary_perfect_training()
    print("✓ test_print_training_summary_perfect_training")

    test_print_training_summary_early_stopping()
    print("✓ test_print_training_summary_early_stopping")

    test_print_training_summary_large_epochs()
    print("✓ test_print_training_summary_large_epochs")

    test_full_training_workflow_output()
    print("✓ test_full_training_workflow_output")

    print("\nAll test_results_printer tests passed!")
