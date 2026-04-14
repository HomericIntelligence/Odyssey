"""Regression tests for ConfusionMatrix.update() dtype validation guard.

Verifies that float labels/predictions are rejected with a descriptive error
rather than silently reinterpreting float bits as integer indices.

Issue: #3686 — ConfusionMatrix.update() silently accepts float32 labels

"""

from std.testing import assert_true, assert_raises
from shared.tensor.any_tensor import AnyTensor
from shared.training.metrics import ConfusionMatrix


def test_float32_labels_raises() raises:
    """Float32 labels must raise a descriptive error."""
    print("Testing float32 labels raise...")

    var cm = ConfusionMatrix(num_classes=3)

    var preds_shape = List[Int]()
    preds_shape.append(2)
    var preds = AnyTensor(preds_shape, DType.int32)
    preds.set(0, Int32(0))
    preds.set(1, Int32(1))

    var labels_shape = List[Int]()
    labels_shape.append(2)
    var labels = AnyTensor(labels_shape, DType.float32)
    labels.set(0, Float32(0.0))
    labels.set(1, Float32(1.0))

    var raised = False
    try:
        cm.update(preds, labels)
    except e:
        raised = True
        var msg = String(e)
        assert_true(
            "int32 or int64" in msg,
            "Error message should mention 'int32 or int64', got: " + msg,
        )

    assert_true(raised, "Expected Error for float32 labels was not raised")
    print("PASS")


def test_float64_labels_raises() raises:
    """Float64 labels must raise a descriptive error."""
    print("Testing float64 labels raise...")

    var cm = ConfusionMatrix(num_classes=3)

    var preds_shape = List[Int]()
    preds_shape.append(2)
    var preds = AnyTensor(preds_shape, DType.int32)
    preds.set(0, Int32(0))
    preds.set(1, Int32(1))

    var labels_shape = List[Int]()
    labels_shape.append(2)
    var labels = AnyTensor(labels_shape, DType.float64)
    labels.set(0, Float64(0.0))
    labels.set(1, Float64(1.0))

    var raised = False
    try:
        cm.update(preds, labels)
    except e:
        raised = True
        var msg = String(e)
        assert_true(
            "int32 or int64" in msg,
            "Error message should mention 'int32 or int64', got: " + msg,
        )

    assert_true(raised, "Expected Error for float64 labels was not raised")
    print("PASS")


def test_float32_predictions_1d_raises() raises:
    """1D float32 predictions must raise a descriptive error."""
    print("Testing 1D float32 predictions raise...")

    var cm = ConfusionMatrix(num_classes=3)

    var preds_shape = List[Int]()
    preds_shape.append(2)
    var preds = AnyTensor(preds_shape, DType.float32)
    preds.set(0, Float32(0.0))
    preds.set(1, Float32(1.0))

    var labels_shape = List[Int]()
    labels_shape.append(2)
    var labels = AnyTensor(labels_shape, DType.int32)
    labels.set(0, Int32(0))
    labels.set(1, Int32(1))

    var raised = False
    try:
        cm.update(preds, labels)
    except e:
        raised = True
        var msg = String(e)
        assert_true(
            "int32 or int64" in msg,
            "Error message should mention 'int32 or int64', got: " + msg,
        )

    assert_true(raised, "Expected Error for float32 predictions was not raised")
    print("PASS")


def test_int32_labels_accepted() raises:
    """Int32 labels must be accepted without error."""
    print("Testing int32 labels accepted...")

    var cm = ConfusionMatrix(num_classes=3)

    var preds_shape = List[Int]()
    preds_shape.append(3)
    var preds = AnyTensor(preds_shape, DType.int32)
    preds.set(0, Int32(0))
    preds.set(1, Int32(1))
    preds.set(2, Int32(2))

    var labels_shape = List[Int]()
    labels_shape.append(3)
    var labels = AnyTensor(labels_shape, DType.int32)
    labels.set(0, Int32(0))
    labels.set(1, Int32(1))
    labels.set(2, Int32(2))

    # Should not raise
    cm.update(preds, labels)
    print("PASS")


def test_int64_labels_accepted() raises:
    """Int64 labels must be accepted without error."""
    print("Testing int64 labels accepted...")

    var cm = ConfusionMatrix(num_classes=3)

    var preds_shape = List[Int]()
    preds_shape.append(3)
    var preds = AnyTensor(preds_shape, DType.int64)
    preds.set(0, Int64(0))
    preds.set(1, Int64(1))
    preds.set(2, Int64(2))

    var labels_shape = List[Int]()
    labels_shape.append(3)
    var labels = AnyTensor(labels_shape, DType.int64)
    labels.set(0, Int64(0))
    labels.set(1, Int64(1))
    labels.set(2, Int64(2))

    # Should not raise
    cm.update(preds, labels)
    print("PASS")


def test_float32_logits_2d_accepted() raises:
    """2D float32 logits (2D predictions) must be accepted (argmax path)."""
    print("Testing 2D float32 logits accepted...")

    var cm = ConfusionMatrix(num_classes=3)

    # 2D logits [2, 3]
    var preds_shape = List[Int]()
    preds_shape.append(2)
    preds_shape.append(3)
    var preds = AnyTensor(preds_shape, DType.float32)
    # Sample 0: logits [1.0, 0.0, 0.0] → argmax=0
    preds.set(0, Float32(1.0))
    preds.set(1, Float32(0.0))
    preds.set(2, Float32(0.0))
    # Sample 1: logits [0.0, 1.0, 0.0] → argmax=1
    preds.set(3, Float32(0.0))
    preds.set(4, Float32(1.0))
    preds.set(5, Float32(0.0))

    var labels_shape = List[Int]()
    labels_shape.append(2)
    var labels = AnyTensor(labels_shape, DType.int32)
    labels.set(0, Int32(0))
    labels.set(1, Int32(1))

    # Should not raise — 2D path goes through argmax which returns int32
    cm.update(preds, labels)
    print("PASS")


def main() raises:
    test_float32_labels_raises()
    test_float64_labels_raises()
    test_float32_predictions_1d_raises()
    test_int32_labels_accepted()
    test_int64_labels_accepted()
    test_float32_logits_2d_accepted()
    print("All dtype guard tests passed!")
