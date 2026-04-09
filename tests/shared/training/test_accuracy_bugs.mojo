"""Tests for List[Int] constructor bugs in accuracy.mojo.

This test file demonstrates the bugs in accuracy.mojo functions that use the
unsafe pattern= List[Int]() followed by list[i] = value.

These tests SHOULD FAIL before the fixes are applied, demonstrating the bug.
After fixing, they should PASS.

Bugs tested:
- Line 118: argmax() - var result_shape = List[Int]()
- Line 348: per_class_accuracy() - var result_shape = List[Int]()
"""

# Import AnyTensor and metrics
from shared.tensor.any_tensor import AnyTensor, ones, zeros
from shared.training.metrics.accuracy import top1_accuracy, per_class_accuracy


# ============================================================================
# Test argmax() bug (Line 118) - called by top1_accuracy
# ============================================================================


def test_top1_accuracy_with_logits() raises:
    """Test top1_accuracy with logits (triggers argmax bug at line 118).

    Bug: Line 118 uses List[Int]() then indexes at line 146.
    This crashes because the list has undefined size.
    """
    # Create logits tensor [batch_size=4, num_classes=3]
    var logits_shape = List[Int]()
    logits_shape.append(4)  # batch size
    logits_shape.append(3)  # num classes
    var logits = ones(logits_shape, DType.float32)

    # Create labels tensor [batch_size=4]
    var labels_shape = List[Int]()
    labels_shape.append(4)
    var labels = zeros(labels_shape, DType.int32)

    # Set some logits to be higher for class 0
    # (simplified - in real test we'd set specific values)

    # This will crash due to bug at line 118 in argmax()
    var acc = top1_accuracy(logits, labels)

    # If we get here, the bug is fixed
    # Accuracy should be > 0 since we're predicting class 0
    print("    Accuracy:", acc)


def test_top1_accuracy_small_batch() raises:
    """Test top1_accuracy with small batch (triggers argmax bug).

    Bug: Line 118 uses List[Int]() then indexes.
    Even with batch_size=1, this crashes.
    """
    # Create logits tensor [batch_size=1, num_classes=10]
    var logits_shape = List[Int]()
    logits_shape.append(1)  # batch size = 1
    logits_shape.append(10)  # num classes
    var logits = ones(logits_shape, DType.float32)

    # Create labels tensor [batch_size=1]
    var labels_shape = List[Int]()
    labels_shape.append(1)
    var labels = zeros(labels_shape, DType.int32)

    # This will crash due to bug at line 118
    var acc = top1_accuracy(logits, labels)

    # If we get here, the bug is fixed
    print("    Accuracy:", acc)


def test_top1_accuracy_large_batch() raises:
    """Test top1_accuracy with larger batch (triggers argmax bug).

    Bug: Line 118 uses List[Int]() then indexes.
    Larger batch size makes the bug more likely to manifest.
    """
    # Create logits tensor [batch_size=100, num_classes=10]
    var logits_shape = List[Int]()
    logits_shape.append(100)  # larger batch
    logits_shape.append(10)  # num classes
    var logits = ones(logits_shape, DType.float32)

    # Create labels tensor [batch_size=100]
    var labels_shape = List[Int]()
    labels_shape.append(100)
    var labels = zeros(labels_shape, DType.int32)

    # This will crash due to bug at line 118
    var acc = top1_accuracy(logits, labels)

    # If we get here, the bug is fixed
    print("    Accuracy:", acc)


# ============================================================================
# Test per_class_accuracy() bug (Line 348)
# ============================================================================


def test_per_class_accuracy_basic() raises:
    """Test per_class_accuracy (triggers bug at line 348).

    Bug: Line 348 uses List[Int]() then indexes at line 358.
    This crashes because the list has undefined size.
    """
    # Create logits tensor [batch_size=10, num_classes=3]
    var logits_shape = List[Int]()
    logits_shape.append(10)  # batch size
    logits_shape.append(3)  # num classes
    var logits = ones(logits_shape, DType.float32)

    # Create labels tensor [batch_size=10]
    var labels_shape = List[Int]()
    labels_shape.append(10)
    var labels = zeros(labels_shape, DType.int32)

    # This will crash due to bug at line 348
    var per_class_acc = per_class_accuracy(logits, labels, num_classes=3)

    # If we get here, the bug is fixed
    print("    Got per-class accuracy tensor")


def test_per_class_accuracy_many_classes() raises:
    """Test per_class_accuracy with many classes (triggers bug at line 348).

    Bug: Line 348 uses List[Int]() then indexes.
    More classes = more likely to crash.
    """
    # Create logits tensor [batch_size=50, num_classes=20]
    var logits_shape = List[Int]()
    logits_shape.append(50)  # batch size
    logits_shape.append(20)  # many classes
    var logits = ones(logits_shape, DType.float32)

    # Create labels tensor [batch_size=50]
    var labels_shape = List[Int]()
    labels_shape.append(50)
    var labels = zeros(labels_shape, DType.int32)

    # This will crash due to bug at line 348
    var per_class_acc = per_class_accuracy(logits, labels, num_classes=20)

    # If we get here, the bug is fixed
    print("    Got per-class accuracy tensor for 20 classes")


def test_per_class_accuracy_few_classes() raises:
    """Test per_class_accuracy with few classes (triggers bug at line 348).

    Bug: Even with num_classes=2, the bug still crashes.
    """
    # Create logits tensor [batch_size=10, num_classes=2]
    var logits_shape = List[Int]()
    logits_shape.append(10)  # batch size
    logits_shape.append(2)  # binary classification
    var logits = ones(logits_shape, DType.float32)

    # Create labels tensor [batch_size=10]
    var labels_shape = List[Int]()
    labels_shape.append(10)
    var labels = zeros(labels_shape, DType.int32)

    # This will crash due to bug at line 348
    var per_class_acc = per_class_accuracy(logits, labels, num_classes=2)

    # If we get here, the bug is fixed
    print("    Got per-class accuracy tensor for binary classification")


# ============================================================================
# Main test runner
# ============================================================================


def main() raises:
    """Run all accuracy.mojo bug tests.

    These tests demonstrate the List[Int] constructor bugs in accuracy.mojo.
    They SHOULD FAIL before fixes are applied, and PASS after.
    """
    print("Running accuracy.mojo List[Int] constructor bug tests...")
    print("WARNING: These tests may crash before fixes are applied!")
    print("")

    # argmax() bugs (via top1_accuracy)
    print("  Testing argmax() bug (via top1_accuracy)...")
    try:
        test_top1_accuracy_with_logits()
        print("    ✓ top1_accuracy with logits")
    except e:
        print("    ✗ top1_accuracy with logits CRASHED:", String(e))

    try:
        test_top1_accuracy_small_batch()
        print("    ✓ top1_accuracy small batch")
    except e:
        print("    ✗ top1_accuracy small batch CRASHED:", String(e))

    try:
        test_top1_accuracy_large_batch()
        print("    ✓ top1_accuracy large batch")
    except e:
        print("    ✗ top1_accuracy large batch CRASHED:", String(e))

    # per_class_accuracy() bugs
    print("  Testing per_class_accuracy() bugs...")
    try:
        test_per_class_accuracy_basic()
        print("    ✓ per_class_accuracy basic")
    except e:
        print("    ✗ per_class_accuracy basic CRASHED:", String(e))

    try:
        test_per_class_accuracy_many_classes()
        print("    ✓ per_class_accuracy many classes")
    except e:
        print("    ✗ per_class_accuracy many classes CRASHED:", String(e))

    try:
        test_per_class_accuracy_few_classes()
        print("    ✓ per_class_accuracy binary")
    except e:
        print("    ✗ per_class_accuracy binary CRASHED:", String(e))

    print("")
    print("accuracy.mojo bug tests completed!")
    print("If any tests crashed, the bugs are still present.")
    print("After fixing, all tests should pass.")
