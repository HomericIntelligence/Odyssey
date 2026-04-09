"""Tests for accuracy metrics.

Comprehensive test suite for top-1, top-k, and per-class accuracy metrics.

Test coverage:
- #279: Accuracy metrics tests

Testing strategy:
- Correctness: Verify accuracy calculations match expected values
- Edge cases: Empty batches, single class, all correct/wrong
- Integration: Test with realistic model outputs
- Incremental: Verify AccuracyMetric accumulation
"""

from testing import assert_true, assert_false, assert_equal, assert_almost_equal
from shared.tensor.any_tensor import AnyTensor
from shared.training.metrics import (
    top1_accuracy,
    topk_accuracy,
    per_class_accuracy,
    AccuracyMetric,
)


def test_top1_accuracy_perfect() raises:
    """Test top-1 accuracy with perfect predictions."""
    print("Testing top-1 accuracy (perfect)...")

    # Create perfect predictions (all correct)
    var batch_size = 10
    var num_classes = 5

    # Logits: make diagonal dominant (correct class has highest score)
    var logits_shape: List[Int] = [batch_size, num_classes]
    var logits = AnyTensor(logits_shape, DType.float32)

    # Labels: 0, 1, 2, 3, 4, 0, 1, 2, 3, 4
    var labels_shape = List[Int]()
    labels_shape.append(batch_size)
    var labels = AnyTensor(labels_shape, DType.int32)

    for i in range(batch_size):
        var true_class = i % num_classes
        labels.set(i, Int32(Int32(true_class)))

        # Set logits: correct class = 10.0, others = 0.0
        for c in range(num_classes):
            var idx = i * num_classes + c
            if c == true_class:
                logits.set(idx, Float32(10.0))
            else:
                logits.set(idx, Float32(0.0))

    var acc = top1_accuracy(logits, labels)

    print("  Accuracy:", acc)
    assert_equal(acc, 1.0, "Perfect predictions should give 100% accuracy")

    print("  ✓ Top-1 accuracy (perfect) test passed")


def test_top1_accuracy_half_correct() raises:
    """Test top-1 accuracy with 50% correct predictions."""
    print("Testing top-1 accuracy (50% correct)...")

    var batch_size = 10
    var num_classes = 3

    var logits_shape: List[Int] = [batch_size, num_classes]
    var logits = AnyTensor(logits_shape, DType.float32)

    var labels_shape = List[Int]()
    labels_shape.append(batch_size)
    var labels = AnyTensor(labels_shape, DType.int32)

    # First 5 correct, last 5 incorrect
    for i in range(batch_size):
        var true_class = i % num_classes
        labels.set(i, Int32(Int32(true_class)))

        if i < 5:
            # Correct prediction
            for c in range(num_classes):
                var idx = i * num_classes + c
                if c == true_class:
                    logits.set(idx, Float32(10.0))
                else:
                    logits.set(idx, Float32(0.0))
        else:
            # Incorrect prediction (predict wrong class)
            var wrong_class = (true_class + 1) % num_classes
            for c in range(num_classes):
                var idx = i * num_classes + c
                if c == wrong_class:
                    logits.set(idx, Float32(10.0))
                else:
                    logits.set(idx, Float32(0.0))

    var acc = top1_accuracy(logits, labels)

    print("  Accuracy:", acc)
    assert_equal(acc, 0.5, "Half correct should give 50% accuracy")

    print("  ✓ Top-1 accuracy (50% correct) test passed")


def test_top1_accuracy_with_indices() raises:
    """Test top-1 accuracy with predicted class indices (not logits)."""
    print("Testing top-1 accuracy with class indices...")

    var batch_size = 8

    # Predicted classes
    var preds_shape = List[Int]()
    preds_shape.append(batch_size)
    var preds = AnyTensor(preds_shape, DType.int32)

    # True labels
    var labels_shape = List[Int]()
    labels_shape.append(batch_size)
    var labels = AnyTensor(labels_shape, DType.int32)

    # Set up: first 4 correct, last 4 incorrect
    for i in range(batch_size):
        labels.set(i, Int32(Int32(i % 4)))

        if i < 4:
            preds.set(i, Int32(Int32(i % 4)))  # Correct
        else:
            preds.set(i, Int32(Int32((i + 1) % 4)))  # Incorrect

    var acc = top1_accuracy(preds, labels)

    print("  Accuracy:", acc)
    assert_equal(acc, 0.5, "4/8 correct should give 50% accuracy")

    print("  ✓ Top-1 accuracy with indices test passed")


def test_topk_accuracy_k1() raises:
    """Test top-k accuracy with k=1 (should match top-1)."""
    print("Testing top-k accuracy (k=1)...")

    var batch_size = 6
    var num_classes = 4

    var logits_shape: List[Int] = [batch_size, num_classes]
    var logits = AnyTensor(logits_shape, DType.float32)

    var labels_shape = List[Int]()
    labels_shape.append(batch_size)
    var labels = AnyTensor(labels_shape, DType.int32)

    # Perfect predictions
    for i in range(batch_size):
        var true_class = i % num_classes
        labels.set(i, Int32(Int32(true_class)))

        for c in range(num_classes):
            var idx = i * num_classes + c
            if c == true_class:
                logits.set(idx, Float32(10.0))
            else:
                logits.set(idx, Float32(c))  # Lower scores

    var acc_top1 = top1_accuracy(logits, labels)
    var acc_topk = topk_accuracy(logits, labels, k=1)

    print("  Top-1 accuracy:", acc_top1)
    print("  Top-k accuracy (k=1):", acc_topk)

    assert_equal(acc_top1, acc_topk, "Top-k with k=1 should match top-1")

    print("  ✓ Top-k accuracy (k=1) test passed")


def test_topk_accuracy_k3() raises:
    """Test top-k accuracy with k=3."""
    print("Testing top-k accuracy (k=3)...")

    var batch_size = 4
    var num_classes = 5

    var logits_shape: List[Int] = [batch_size, num_classes]
    var logits = AnyTensor(logits_shape, DType.float32)

    var labels_shape = List[Int]()
    labels_shape.append(batch_size)
    var labels = AnyTensor(labels_shape, DType.int32)

    # Sample 0: true=0, scores=[5, 4, 3, 2, 1] -> top-3=[0,1,2] -> correct
    # Sample 1: true=1, scores=[1, 2, 5, 4, 3] -> top-3=[2,3,4] -> incorrect
    # Sample 2: true=2, scores=[3, 4, 5, 1, 2] -> top-3=[2,1,0] -> correct
    # Sample 3: true=4, scores=[5, 4, 3, 2, 1] -> top-3=[0,1,2] -> incorrect
    # Expected: 2/4 = 0.5

    # Create 2D array of scores using direct indexing
    var scores = List[List[Float32]]()

    # Sample 0
    var s0 = List[Float32]()
    s0.append(5.0)
    s0.append(4.0)
    s0.append(3.0)
    s0.append(2.0)
    s0.append(1.0)
    scores.append(s0^)

    # Sample 1
    var s1 = List[Float32]()
    s1.append(1.0)
    s1.append(2.0)
    s1.append(5.0)
    s1.append(4.0)
    s1.append(3.0)
    scores.append(s1^)

    # Sample 2
    var s2 = List[Float32]()
    s2.append(3.0)
    s2.append(4.0)
    s2.append(5.0)
    s2.append(1.0)
    s2.append(2.0)
    scores.append(s2^)

    # Sample 3
    var s3 = List[Float32]()
    s3.append(5.0)
    s3.append(4.0)
    s3.append(3.0)
    s3.append(2.0)
    s3.append(1.0)
    scores.append(s3^)

    # Fill logits
    for i in range(batch_size):
        for c in range(num_classes):
            var idx = i * num_classes + c
            logits.set(idx, Float32(scores[i][c]))

    # Set labels
    labels.set(0, Int32(0))
    labels.set(1, Int32(1))
    labels.set(2, Int32(2))
    labels.set(3, Int32(4))

    var acc = topk_accuracy(logits, labels, k=3)

    print("  Top-3 accuracy:", acc)
    assert_equal(acc, 0.5, "2 out of 4 in top-3 should give 50%")

    print("  ✓ Top-k accuracy (k=3) test passed")


def test_per_class_accuracy() raises:
    """Test per-class accuracy computation."""
    print("Testing per-class accuracy...")

    var batch_size = 12
    var num_classes = 3

    var logits_shape: List[Int] = [batch_size, num_classes]
    var logits = AnyTensor(logits_shape, DType.float32)

    var labels_shape = List[Int]()
    labels_shape.append(batch_size)
    var labels = AnyTensor(labels_shape, DType.int32)

    # Class 0: 4 samples, 3 correct -> 75%
    # Class 1: 4 samples, 2 correct -> 50%
    # Class 2: 4 samples, 4 correct -> 100%

    var true_classes = List[Int]()
    true_classes.append(0)
    true_classes.append(0)
    true_classes.append(0)
    true_classes.append(0)
    true_classes.append(1)
    true_classes.append(1)
    true_classes.append(1)
    true_classes.append(1)
    true_classes.append(2)
    true_classes.append(2)
    true_classes.append(2)
    true_classes.append(2)

    var pred_classes = List[Int]()
    pred_classes.append(0)
    pred_classes.append(0)
    pred_classes.append(0)
    pred_classes.append(1)  # 3/4 for class 0
    pred_classes.append(1)
    pred_classes.append(1)
    pred_classes.append(0)
    pred_classes.append(2)  # 2/4 for class 1
    pred_classes.append(2)
    pred_classes.append(2)
    pred_classes.append(2)
    pred_classes.append(2)  # 4/4 for class 2

    # Fill data
    for i in range(batch_size):
        labels.set(i, Int32(Int32(true_classes[i])))

        # Make predicted class have highest score
        for c in range(num_classes):
            var idx = i * num_classes + c
            if c == pred_classes[i]:
                logits.set(idx, Float32(10.0))
            else:
                logits.set(idx, Float32(0.0))

    var per_class_acc = per_class_accuracy(logits, labels, num_classes)

    var acc_class0 = per_class_acc._data.bitcast[Float64]()[0]
    var acc_class1 = per_class_acc._data.bitcast[Float64]()[1]
    var acc_class2 = per_class_acc._data.bitcast[Float64]()[2]

    print("  Class 0 accuracy:", acc_class0)
    print("  Class 1 accuracy:", acc_class1)
    print("  Class 2 accuracy:", acc_class2)

    assert_equal(acc_class0, 0.75, "Class 0 should have 75% accuracy")
    assert_equal(acc_class1, 0.5, "Class 1 should have 50% accuracy")
    assert_equal(acc_class2, 1.0, "Class 2 should have 100% accuracy")

    print("  ✓ Per-class accuracy test passed")


def test_accuracy_metric_incremental() raises:
    """Test AccuracyMetric incremental accumulation."""
    print("Testing AccuracyMetric incremental...")

    var metric = AccuracyMetric()

    # Batch 1: 8 samples, 6 correct
    var batch1_size = 8
    var logits1_shape: List[Int] = [batch1_size, 3]
    var logits1 = AnyTensor(logits1_shape, DType.float32)
    var labels1_shape = List[Int]()
    labels1_shape.append(batch1_size)
    var labels1 = AnyTensor(labels1_shape, DType.int32)

    for i in range(batch1_size):
        var true_class = i % 3
        labels1.set(i, Int32(Int32(true_class)))

        # First 6 correct, last 2 incorrect
        var pred_class = true_class if i < 6 else (true_class + 1) % 3

        for c in range(3):
            var idx = i * 3 + c
            if c == pred_class:
                logits1.set(idx, Float32(10.0))
            else:
                logits1.set(idx, Float32(0.0))

    metric.update(logits1, labels1)

    # After batch 1: 6/8 correct
    var acc1 = metric.compute()
    print("  After batch 1:", acc1)
    assert_equal(acc1, 0.75, "First batch should have 75% accuracy")

    # Batch 2: 4 samples, 2 correct
    var batch2_size = 4
    var logits2_shape: List[Int] = [batch2_size, 3]
    var logits2 = AnyTensor(logits2_shape, DType.float32)
    var labels2_shape = List[Int]()
    labels2_shape.append(batch2_size)
    var labels2 = AnyTensor(labels2_shape, DType.int32)

    for i in range(batch2_size):
        var true_class = i % 3
        labels2.set(i, Int32(Int32(true_class)))

        # First 2 correct, last 2 incorrect
        var pred_class = true_class if i < 2 else (true_class + 1) % 3

        for c in range(3):
            var idx = i * 3 + c
            if c == pred_class:
                logits2.set(idx, Float32(10.0))
            else:
                logits2.set(idx, Float32(0.0))

    metric.update(logits2, labels2)

    # After batch 2: (6+2)/(8+4) = 8/12 = 0.6667
    var acc2 = metric.compute()
    print("  After batch 2:", acc2)

    var expected_acc = 8.0 / 12.0
    var diff = abs(acc2 - expected_acc)
    assert_true(diff < 0.001, "Accumulated accuracy should be 8/12")

    # Reset and verify
    metric.reset()
    var acc_after_reset = metric.compute()
    assert_equal(acc_after_reset, 0.0, "After reset should be 0")

    print("  ✓ AccuracyMetric incremental test passed")


def test_accuracy_metric_empty() raises:
    """Test AccuracyMetric with no data."""
    print("Testing AccuracyMetric empty...")

    var metric = AccuracyMetric()
    var acc = metric.compute()

    assert_equal(acc, 0.0, "Empty metric should return 0.0")

    print("  ✓ AccuracyMetric empty test passed")


def main() raises:
    """Run all accuracy metric tests."""
    print("\n" + "=" * 70)
    print("ACCURACY METRICS TEST SUITE")
    print("=" * 70 + "\n")

    print("Top-1 Accuracy Tests (#279)")
    print("-" * 70)
    test_top1_accuracy_perfect()
    test_top1_accuracy_half_correct()
    test_top1_accuracy_with_indices()

    print("\nTop-K Accuracy Tests (#279)")
    print("-" * 70)
    test_topk_accuracy_k1()
    test_topk_accuracy_k3()

    print("\nPer-Class Accuracy Tests (#279)")
    print("-" * 70)
    test_per_class_accuracy()

    print("\nIncremental Metric Tests (#279)")
    print("-" * 70)
    test_accuracy_metric_incremental()
    test_accuracy_metric_empty()

    print("\n" + "=" * 70)
    print("ALL ACCURACY METRICS TESTS PASSED ✓")
    print("=" * 70 + "\n")
