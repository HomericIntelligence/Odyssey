"""Tests for confusion matrix metrics (Part 2 of 2).

Edge case and logits tests for ConfusionMatrix.

Test coverage:
- #289: Confusion matrix tests (part 2)

Testing strategy:
- Logits: Test with 2D logit inputs (argmax-based prediction)
- Edge cases: Empty matrix, reset functionality

ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
high test load. Split from test_confusion_matrix.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md

Note: Split from monolithic test file due to Mojo 0.26.1 heap corruption
bug that occurs after ~15 cumulative tests. See Issue #2942.
"""

from testing import assert_true, assert_false, assert_equal, assert_almost_equal
from shared.core.extensor import ExTensor
from shared.training.metrics import ConfusionMatrix


fn test_confusion_matrix_with_logits() raises:
    """Test confusion matrix with logits (not class indices)."""
    print("Testing ConfusionMatrix with logits...")

    var cm = ConfusionMatrix(num_classes=3)

    # Create logits [batch_size=4, num_classes=3]
    var logits_shape = List[Int]()
    logits_shape.append(4)
    logits_shape.append(3)
    var logits = ExTensor(logits_shape, DType.float32)
    var labels_shape = List[Int]()
    labels_shape.append(4)  # 4 samples
    var labels = ExTensor(labels_shape, DType.int32)

    # Sample 0: true=0, logits=[10, 0, 0] -> pred=0 ✓
    logits._data.bitcast[Float32]()[0] = 10.0
    logits._data.bitcast[Float32]()[1] = 0.0
    logits._data.bitcast[Float32]()[2] = 0.0
    labels._data.bitcast[Int32]()[0] = 0

    # Sample 1: true=1, logits=[0, 10, 0] -> pred=1 ✓
    logits._data.bitcast[Float32]()[3] = 0.0
    logits._data.bitcast[Float32]()[4] = 10.0
    logits._data.bitcast[Float32]()[5] = 0.0
    labels._data.bitcast[Int32]()[1] = 1

    # Sample 2: true=2, logits=[0, 0, 10] -> pred=2 ✓
    logits._data.bitcast[Float32]()[6] = 0.0
    logits._data.bitcast[Float32]()[7] = 0.0
    logits._data.bitcast[Float32]()[8] = 10.0
    labels._data.bitcast[Int32]()[2] = 2

    # Sample 3: true=0, logits=[0, 10, 0] -> pred=1 ✗
    logits._data.bitcast[Float32]()[9] = 0.0
    logits._data.bitcast[Float32]()[10] = 10.0
    logits._data.bitcast[Float32]()[11] = 0.0
    labels._data.bitcast[Int32]()[3] = 0

    cm.update(logits, labels)

    # Expected matrix: [1,1,0; 0,1,0; 0,0,1]
    var raw = cm.normalize(mode="none")

    assert_equal(
        Int(raw._data.bitcast[Float64]()[0]), 1, "Matrix[0,0] should be 1"
    )
    assert_equal(
        Int(raw._data.bitcast[Float64]()[1]), 1, "Matrix[0,1] should be 1"
    )
    assert_equal(
        Int(raw._data.bitcast[Float64]()[4]), 1, "Matrix[1,1] should be 1"
    )
    assert_equal(
        Int(raw._data.bitcast[Float64]()[8]), 1, "Matrix[2,2] should be 1"
    )

    print("  ✓ Logits test passed")


fn test_confusion_matrix_reset() raises:
    """Test resetting confusion matrix."""
    print("Testing ConfusionMatrix reset...")

    var cm = ConfusionMatrix(num_classes=2)

    # Add some data
    var preds_shape = List[Int]()
    preds_shape.append(2)  # 2 samples
    var preds = ExTensor(preds_shape, DType.int32)
    var labels_shape = List[Int]()
    labels_shape.append(2)  # 2 samples
    var labels = ExTensor(labels_shape, DType.int32)
    preds._data.bitcast[Int32]()[0] = 0
    preds._data.bitcast[Int32]()[1] = 1
    labels._data.bitcast[Int32]()[0] = 0
    labels._data.bitcast[Int32]()[1] = 1

    cm.update(preds, labels)

    # Reset
    cm.reset()

    # All values should be 0
    var raw = cm.normalize(mode="none")
    for i in range(4):
        assert_equal(
            Int(raw._data.bitcast[Float64]()[i]),
            0,
            "All values should be 0 after reset",
        )

    print("  ✓ Reset test passed")


fn test_confusion_matrix_empty() raises:
    """Test confusion matrix with no data."""
    print("Testing ConfusionMatrix empty...")

    var cm = ConfusionMatrix(num_classes=3)

    # Get metrics without adding data
    var precision = cm.get_precision()
    var recall = cm.get_recall()
    var f1 = cm.get_f1_score()

    # All should be 0.0
    for i in range(3):
        assert_equal(
            precision._data.bitcast[Float64]()[i],
            0.0,
            "Empty precision should be 0.0",
        )
        assert_equal(
            recall._data.bitcast[Float64]()[i],
            0.0,
            "Empty recall should be 0.0",
        )
        assert_equal(
            f1._data.bitcast[Float64]()[i], 0.0, "Empty F1 should be 0.0"
        )

    print("  ✓ Empty matrix test passed")


fn test_confusion_matrix_single_class() raises:
    """Test confusion matrix with all predictions in one class. Closes #3686."""
    print("Testing ConfusionMatrix single class predictions...")

    var cm = ConfusionMatrix(num_classes=3)

    # All predictions are class 0, all labels are class 0
    var preds_shape = List[Int]()
    preds_shape.append(3)
    var preds = ExTensor(preds_shape, DType.int32)
    var labels = ExTensor(preds_shape, DType.int32)
    preds._data.bitcast[Int32]()[0] = 0
    preds._data.bitcast[Int32]()[1] = 0
    preds._data.bitcast[Int32]()[2] = 0
    labels._data.bitcast[Int32]()[0] = 0
    labels._data.bitcast[Int32]()[1] = 0
    labels._data.bitcast[Int32]()[2] = 0

    cm.update(preds, labels)

    var raw = cm.normalize(mode="none")
    assert_equal(
        Int(raw._data.bitcast[Float64]()[0]),
        3,
        "Matrix[0,0] should be 3",
    )

    print("  ✓ Single class test passed")


fn test_confusion_matrix_misclassification() raises:
    """Test confusion matrix captures misclassifications correctly."""
    print("Testing ConfusionMatrix misclassification...")

    var cm = ConfusionMatrix(num_classes=2)

    var shape = List[Int]()
    shape.append(4)
    var preds = ExTensor(shape, DType.int32)
    var labels = ExTensor(shape, DType.int32)

    # 2 correct, 2 wrong
    preds._data.bitcast[Int32]()[0] = 0
    labels._data.bitcast[Int32]()[0] = 0  # correct
    preds._data.bitcast[Int32]()[1] = 1
    labels._data.bitcast[Int32]()[1] = 1  # correct
    preds._data.bitcast[Int32]()[2] = 0
    labels._data.bitcast[Int32]()[2] = 1  # wrong
    preds._data.bitcast[Int32]()[3] = 1
    labels._data.bitcast[Int32]()[3] = 0  # wrong

    cm.update(preds, labels)

    var raw = cm.normalize(mode="none")
    # [1, 1; 1, 1]
    assert_equal(
        Int(raw._data.bitcast[Float64]()[0]),
        1,
        "Matrix[0,0] should be 1",
    )
    assert_equal(
        Int(raw._data.bitcast[Float64]()[1]),
        1,
        "Matrix[0,1] should be 1",
    )
    assert_equal(
        Int(raw._data.bitcast[Float64]()[2]),
        1,
        "Matrix[1,0] should be 1",
    )
    assert_equal(
        Int(raw._data.bitcast[Float64]()[3]),
        1,
        "Matrix[1,1] should be 1",
    )

    print("  ✓ Misclassification test passed")


fn main() raises:
    """Run confusion matrix tests (Part 2): logits and edge cases."""
    print("\n" + "=" * 70)
    print("CONFUSION MATRIX TEST SUITE - PART 2")
    print("=" * 70 + "\n")

    print("Logits Input Tests (#289)")
    print("-" * 70)
    test_confusion_matrix_with_logits()

    print("\nEdge Cases (#289)")
    print("-" * 70)
    test_confusion_matrix_reset()
    test_confusion_matrix_empty()

    print("\nAdditional Tests (#3686)")
    print("-" * 70)
    test_confusion_matrix_single_class()
    test_confusion_matrix_misclassification()

    print("\n" + "=" * 70)
    print("ALL CONFUSION MATRIX PART 2 TESTS PASSED ✓")
    print("=" * 70 + "\n")
