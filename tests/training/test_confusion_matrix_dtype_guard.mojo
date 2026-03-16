"""Regression tests for ConfusionMatrix.update() dtype validation guard.

Verifies that float labels/predictions are rejected with a descriptive error
rather than silently reinterpreting float bits as integer indices.

Issue: #3686 — ConfusionMatrix.update() silently accepts float32 labels

ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
high test load. See docs/adr/ADR-009-heap-corruption-workaround.md
"""

from testing import assert_true, assert_raises
from shared.core import ExTensor
from shared.training.metrics import ConfusionMatrix


fn test_float32_labels_raises() raises:
    """Float32 labels must raise a descriptive error."""
    print("Testing float32 labels raise...")

    var cm = ConfusionMatrix(num_classes=3)

    var preds_shape = List[Int]()
    preds_shape.append(2)
    var preds = ExTensor(preds_shape, DType.int32)
    preds._data.bitcast[Int32]()[0] = 0
    preds._data.bitcast[Int32]()[1] = 1

    var labels_shape = List[Int]()
    labels_shape.append(2)
    var labels = ExTensor(labels_shape, DType.float32)
    labels._data.bitcast[Float32]()[0] = 0.0
    labels._data.bitcast[Float32]()[1] = 1.0

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


fn test_float64_labels_raises() raises:
    """Float64 labels must raise a descriptive error."""
    print("Testing float64 labels raise...")

    var cm = ConfusionMatrix(num_classes=3)

    var preds_shape = List[Int]()
    preds_shape.append(2)
    var preds = ExTensor(preds_shape, DType.int32)
    preds._data.bitcast[Int32]()[0] = 0
    preds._data.bitcast[Int32]()[1] = 1

    var labels_shape = List[Int]()
    labels_shape.append(2)
    var labels = ExTensor(labels_shape, DType.float64)
    labels._data.bitcast[Float64]()[0] = 0.0
    labels._data.bitcast[Float64]()[1] = 1.0

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


fn test_float32_predictions_1d_raises() raises:
    """1D float32 predictions must raise a descriptive error."""
    print("Testing 1D float32 predictions raise...")

    var cm = ConfusionMatrix(num_classes=3)

    var preds_shape = List[Int]()
    preds_shape.append(2)
    var preds = ExTensor(preds_shape, DType.float32)
    preds._data.bitcast[Float32]()[0] = 0.0
    preds._data.bitcast[Float32]()[1] = 1.0

    var labels_shape = List[Int]()
    labels_shape.append(2)
    var labels = ExTensor(labels_shape, DType.int32)
    labels._data.bitcast[Int32]()[0] = 0
    labels._data.bitcast[Int32]()[1] = 1

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


fn test_int32_labels_accepted() raises:
    """Int32 labels must be accepted without error."""
    print("Testing int32 labels accepted...")

    var cm = ConfusionMatrix(num_classes=3)

    var preds_shape = List[Int]()
    preds_shape.append(3)
    var preds = ExTensor(preds_shape, DType.int32)
    preds._data.bitcast[Int32]()[0] = 0
    preds._data.bitcast[Int32]()[1] = 1
    preds._data.bitcast[Int32]()[2] = 2

    var labels_shape = List[Int]()
    labels_shape.append(3)
    var labels = ExTensor(labels_shape, DType.int32)
    labels._data.bitcast[Int32]()[0] = 0
    labels._data.bitcast[Int32]()[1] = 1
    labels._data.bitcast[Int32]()[2] = 2

    # Should not raise
    cm.update(preds, labels)
    print("PASS")


fn test_int64_labels_accepted() raises:
    """Int64 labels must be accepted without error."""
    print("Testing int64 labels accepted...")

    var cm = ConfusionMatrix(num_classes=3)

    var preds_shape = List[Int]()
    preds_shape.append(3)
    var preds = ExTensor(preds_shape, DType.int64)
    preds._data.bitcast[Int64]()[0] = 0
    preds._data.bitcast[Int64]()[1] = 1
    preds._data.bitcast[Int64]()[2] = 2

    var labels_shape = List[Int]()
    labels_shape.append(3)
    var labels = ExTensor(labels_shape, DType.int64)
    labels._data.bitcast[Int64]()[0] = 0
    labels._data.bitcast[Int64]()[1] = 1
    labels._data.bitcast[Int64]()[2] = 2

    # Should not raise
    cm.update(preds, labels)
    print("PASS")


fn test_float32_logits_2d_accepted() raises:
    """2D float32 logits (2D predictions) must be accepted (argmax path)."""
    print("Testing 2D float32 logits accepted...")

    var cm = ConfusionMatrix(num_classes=3)

    # 2D logits [2, 3]
    var preds_shape = List[Int]()
    preds_shape.append(2)
    preds_shape.append(3)
    var preds = ExTensor(preds_shape, DType.float32)
    # Sample 0: logits [1.0, 0.0, 0.0] → argmax=0
    preds._data.bitcast[Float32]()[0] = 1.0
    preds._data.bitcast[Float32]()[1] = 0.0
    preds._data.bitcast[Float32]()[2] = 0.0
    # Sample 1: logits [0.0, 1.0, 0.0] → argmax=1
    preds._data.bitcast[Float32]()[3] = 0.0
    preds._data.bitcast[Float32]()[4] = 1.0
    preds._data.bitcast[Float32]()[5] = 0.0

    var labels_shape = List[Int]()
    labels_shape.append(2)
    var labels = ExTensor(labels_shape, DType.int32)
    labels._data.bitcast[Int32]()[0] = 0
    labels._data.bitcast[Int32]()[1] = 1

    # Should not raise — 2D path goes through argmax which returns int32
    cm.update(preds, labels)
    print("PASS")


fn main() raises:
    test_float32_labels_raises()
    test_float64_labels_raises()
    test_float32_predictions_1d_raises()
    test_int32_labels_accepted()
    test_int64_labels_accepted()
    test_float32_logits_2d_accepted()
    print("All dtype guard tests passed!")
