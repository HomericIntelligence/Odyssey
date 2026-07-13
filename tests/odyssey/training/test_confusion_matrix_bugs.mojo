"""Tests for List[Int] constructor bugs in confusion_matrix.mojo.

This test file demonstrates the bug in confusion_matrix.mojo that uses the
unsafe pattern= List[Int]() followed by list[i] = value.

These tests SHOULD FAIL before the fix is applied, demonstrating the bug.
After fixing, they should PASS.

Bug tested:
- Line 323: argmax() - var result_shape = List[Int]()
"""

# Import AnyTensor and confusion matrix
from odyssey.tensor.any_tensor import AnyTensor
from odyssey.tensor.tensor_creation import ones, zeros
from odyssey.training.metrics.confusion_matrix import ConfusionMatrix


# ============================================================================
# Test argmax() bug (Line 323) - called by ConfusionMatrix.update()
# ============================================================================


def test_confusion_matrix_update_with_logits() raises:
    """Test ConfusionMatrix.update with logits (triggers argmax bug at line 323).

    Bug: Line 323 uses List[Int]() then indexes at line 350.
    This crashes because the list has undefined size.
    """
    # Create confusion matrix for 3 classes
    var cm = ConfusionMatrix(num_classes=3)

    # Create logits tensor [batch_size=10, num_classes=3]
    var logits_shape = List[Int]()
    logits_shape.append(10)  # batch size
    logits_shape.append(3)  # num classes
    var logits = ones(logits_shape, DType.float32)

    # Create labels tensor [batch_size=10]
    var labels_shape = List[Int]()
    labels_shape.append(10)
    var labels = zeros(labels_shape, DType.int32)

    # This will crash due to bug at line 323 in argmax()
    cm.update(logits, labels)

    # If we get here, the bug is fixed
    print("    ✓ ConfusionMatrix updated successfully")


def test_confusion_matrix_small_batch() raises:
    """Test ConfusionMatrix with small batch (triggers argmax bug).

    Bug: Line 323 uses List[Int]() then indexes.
    Even with batch_size=1, this crashes.
    """
    # Create confusion matrix for 5 classes
    var cm = ConfusionMatrix(num_classes=5)

    # Create logits tensor [batch_size=1, num_classes=5]
    var logits_shape = List[Int]()
    logits_shape.append(1)  # batch size = 1
    logits_shape.append(5)  # num classes
    var logits = ones(logits_shape, DType.float32)

    # Create labels tensor [batch_size=1]
    var labels_shape = List[Int]()
    labels_shape.append(1)
    var labels = zeros(labels_shape, DType.int32)

    # This will crash due to bug at line 323
    cm.update(logits, labels)

    # If we get here, the bug is fixed
    print("    ✓ ConfusionMatrix small batch updated")


def test_confusion_matrix_large_batch() raises:
    """Test ConfusionMatrix with large batch (triggers argmax bug).

    Bug: Line 323 uses List[Int]() then indexes.
    Larger batch makes bug more likely to manifest.
    """
    # Create confusion matrix for 10 classes
    var cm = ConfusionMatrix(num_classes=10)

    # Create logits tensor [batch_size=128, num_classes=10]
    var logits_shape = List[Int]()
    logits_shape.append(128)  # large batch
    logits_shape.append(10)  # num classes
    var logits = ones(logits_shape, DType.float32)

    # Create labels tensor [batch_size=128]
    var labels_shape = List[Int]()
    labels_shape.append(128)
    var labels = zeros(labels_shape, DType.int32)

    # This will crash due to bug at line 323
    cm.update(logits, labels)

    # If we get here, the bug is fixed
    print("    ✓ ConfusionMatrix large batch updated")


def test_confusion_matrix_multiple_updates() raises:
    """Test ConfusionMatrix with multiple update calls (triggers bug repeatedly).

    Bug: Each call to update() with logits triggers the argmax bug.
    """
    # Create confusion matrix for 4 classes
    var cm = ConfusionMatrix(num_classes=4)

    # Update with multiple batches
    for _ in range(3):
        # Create logits tensor [batch_size=16, num_classes=4]
        var logits_shape = List[Int]()
        logits_shape.append(16)
        logits_shape.append(4)
        var logits = ones(logits_shape, DType.float32)

        # Create labels tensor [batch_size=16]
        var labels_shape = List[Int]()
        labels_shape.append(16)
        var labels = zeros(labels_shape, DType.int32)

        # This will crash due to bug at line 323
        cm.update(logits, labels)

    # If we get here, the bug is fixed
    print("    ✓ ConfusionMatrix multiple updates completed")


def test_confusion_matrix_binary_classification() raises:
    """Test ConfusionMatrix for binary classification (triggers bug).

    Bug: Even with just 2 classes, the bug still crashes.
    """
    # Create confusion matrix for binary classification
    var cm = ConfusionMatrix(num_classes=2)

    # Create logits tensor [batch_size=20, num_classes=2]
    var logits_shape = List[Int]()
    logits_shape.append(20)
    logits_shape.append(2)
    var logits = ones(logits_shape, DType.float32)

    # Create labels tensor [batch_size=20]
    var labels_shape = List[Int]()
    labels_shape.append(20)
    var labels = zeros(labels_shape, DType.int32)

    # This will crash due to bug at line 323
    cm.update(logits, labels)

    # If we get here, the bug is fixed
    print("    ✓ ConfusionMatrix binary classification updated")


# ============================================================================
# Main test runner
# ============================================================================


def main() raises:
    """Run all confusion_matrix.mojo bug tests.

    These tests demonstrate the List[Int] constructor bug in confusion_matrix.mojo.
    They SHOULD FAIL before fix is applied, and PASS after.
    """
    print("Running confusion_matrix.mojo List[Int] constructor bug tests...")
    print("WARNING: These tests may crash before fix is applied!")
    print("")

    print("  Testing ConfusionMatrix.update() argmax bug...")
    try:
        test_confusion_matrix_update_with_logits()
    except e:
        print("    ✗ update with logits CRASHED:", String(e))

    try:
        test_confusion_matrix_small_batch()
    except e:
        print("    ✗ small batch CRASHED:", String(e))

    try:
        test_confusion_matrix_large_batch()
    except e:
        print("    ✗ large batch CRASHED:", String(e))

    try:
        test_confusion_matrix_multiple_updates()
    except e:
        print("    ✗ multiple updates CRASHED:", String(e))

    try:
        test_confusion_matrix_binary_classification()
    except e:
        print("    ✗ binary classification CRASHED:", String(e))

    print("")
    print("confusion_matrix.mojo bug tests completed!")
    print("If any tests crashed, the bug is still present.")
    print("After fixing, all tests should pass.")
