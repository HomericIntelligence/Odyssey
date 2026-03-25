"""Tests for confusion matrix metrics

Basic functionality, normalization, and derived metrics tests for ConfusionMatrix.

Test coverage:
- #289: Confusion matrix tests (part 1)

Testing strategy:
- Correctness: Verify matrix values and derived metrics
- Normalization: Test all modes (row, column, total, none)
- Multi-class: Test with various class distributions

"""


from testing import assert_true, assert_false, assert_equal, assert_almost_equal
from shared.tensor.any_tensor import AnyTensor
from shared.training.metrics import ConfusionMatrix


fn test_confusion_matrix_basic() raises:
    """Test basic confusion matrix functionality."""
    print("Testing ConfusionMatrix basic...")

    var cm = ConfusionMatrix(num_classes=3)

    # Create predictions and labels
    # Sample 0: true=0, pred=0 ✓
    # Sample 1: true=1, pred=1 ✓
    # Sample 2: true=2, pred=2 ✓
    # Sample 3: true=0, pred=1 ✗
    # Sample 4: true=1, pred=2 ✗

    var preds_shape = List[Int]()
    preds_shape.append(5)  # 5 samples
    var preds = AnyTensor(preds_shape, DType.int32)
    preds._data.bitcast[Int32]()[0] = 0
    preds._data.bitcast[Int32]()[1] = 1
    preds._data.bitcast[Int32]()[2] = 2
    preds._data.bitcast[Int32]()[3] = 1  # Wrong
    preds._data.bitcast[Int32]()[4] = 2  # Wrong

    var labels_shape = List[Int]()
    labels_shape.append(5)  # 5 samples
    var labels = AnyTensor(labels_shape, DType.int32)
    labels._data.bitcast[Int32]()[0] = 0
    labels._data.bitcast[Int32]()[1] = 1
    labels._data.bitcast[Int32]()[2] = 2
    labels._data.bitcast[Int32]()[3] = 0
    labels._data.bitcast[Int32]()[4] = 1

    cm.update(preds, labels)

    # Expected matrix:
    # [1 1 0]  (true class 0: 1 correct as 0, 1 wrong as 1)
    # [0 1 1]  (true class 1: 1 correct as 1, 1 wrong as 2)
    # [0 0 1]  (true class 2: 1 correct as 2)

    var raw = cm.normalize(mode="none")

    print("  Confusion matrix (raw counts):")
    for i in range(3):
        var row_str = "    ["
        for j in range(3):
            var idx = i * 3 + j
            var val = Int(raw._data.bitcast[Float64]()[idx])
            row_str = row_str + String(val)
            if j < 2:
                row_str = row_str + " "
        row_str = row_str + "]"
        print(row_str)

    # Verify specific values
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
        Int(raw._data.bitcast[Float64]()[5]), 1, "Matrix[1,2] should be 1"
    )
    assert_equal(
        Int(raw._data.bitcast[Float64]()[8]), 1, "Matrix[2,2] should be 1"
    )

    print("  ✓ ConfusionMatrix basic test passed")


fn test_confusion_matrix_perfect() raises:
    """Test confusion matrix with perfect predictions."""
    print("Testing ConfusionMatrix perfect predictions...")

    var cm = ConfusionMatrix(num_classes=3)

    # All predictions correct
    var preds_shape = List[Int]()
    preds_shape.append(6)  # 6 samples
    var preds = AnyTensor(preds_shape, DType.int32)
    var labels_shape = List[Int]()
    labels_shape.append(6)  # 6 samples
    var labels = AnyTensor(labels_shape, DType.int32)

    for i in range(6):
        var cls = i % 3
        preds._data.bitcast[Int32]()[i] = Int32(cls)
        labels._data.bitcast[Int32]()[i] = Int32(cls)

    cm.update(preds, labels)

    # Expected: diagonal matrix [2,0,0; 0,2,0; 0,0,2]
    var raw = cm.normalize(mode="none")

    assert_equal(
        Int(raw._data.bitcast[Float64]()[0]), 2, "Matrix[0,0] should be 2"
    )
    assert_equal(
        Int(raw._data.bitcast[Float64]()[4]), 2, "Matrix[1,1] should be 2"
    )
    assert_equal(
        Int(raw._data.bitcast[Float64]()[8]), 2, "Matrix[2,2] should be 2"
    )

    # All off-diagonal should be 0
    for i in range(3):
        for j in range(3):
            if i != j:
                var idx = i * 3 + j
                assert_equal(
                    Int(raw._data.bitcast[Float64]()[idx]),
                    0,
                    "Off-diagonal should be 0",
                )

    # Precision, recall, F1 should all be 1.0
    var precision = cm.get_precision()
    var recall = cm.get_recall()
    var f1 = cm.get_f1_score()

    for i in range(3):
        assert_equal(
            precision._data.bitcast[Float64]()[i],
            1.0,
            "Precision should be 1.0",
        )
        assert_equal(
            recall._data.bitcast[Float64]()[i], 1.0, "Recall should be 1.0"
        )
        assert_equal(f1._data.bitcast[Float64]()[i], 1.0, "F1 should be 1.0")

    print("  ✓ Perfect predictions test passed")


fn test_confusion_matrix_normalize_row() raises:
    """Test row normalization (recall per class)."""
    print("Testing ConfusionMatrix row normalization...")

    var cm = ConfusionMatrix(num_classes=2)

    # Class 0: 2 samples, 1 correct (50%)
    # Class 1: 2 samples, 2 correct (100%)
    var preds_shape = List[Int]()
    preds_shape.append(4)  # 4 samples
    var preds = AnyTensor(preds_shape, DType.int32)
    var labels_shape = List[Int]()
    labels_shape.append(4)  # 4 samples
    var labels = AnyTensor(labels_shape, DType.int32)

    preds._data.bitcast[Int32]()[0] = 0  # ✓
    preds._data.bitcast[Int32]()[1] = 1  # ✗ (true=0)
    preds._data.bitcast[Int32]()[2] = 1  # ✓
    preds._data.bitcast[Int32]()[3] = 1  # ✓

    labels._data.bitcast[Int32]()[0] = 0
    labels._data.bitcast[Int32]()[1] = 0
    labels._data.bitcast[Int32]()[2] = 1
    labels._data.bitcast[Int32]()[3] = 1

    cm.update(preds, labels)

    var row_norm = cm.normalize(mode="row")

    # Row 0: [1, 1] -> [0.5, 0.5]
    # Row 1: [0, 2] -> [0.0, 1.0]

    assert_equal(
        row_norm._data.bitcast[Float64]()[0], 0.5, "Row 0, col 0 should be 0.5"
    )
    assert_equal(
        row_norm._data.bitcast[Float64]()[1], 0.5, "Row 0, col 1 should be 0.5"
    )
    assert_equal(
        row_norm._data.bitcast[Float64]()[2], 0.0, "Row 1, col 0 should be 0.0"
    )
    assert_equal(
        row_norm._data.bitcast[Float64]()[3], 1.0, "Row 1, col 1 should be 1.0"
    )

    print("  ✓ Row normalization test passed")


fn test_confusion_matrix_normalize_column() raises:
    """Test column normalization (precision per class)."""
    print("Testing ConfusionMatrix column normalization...")

    var cm = ConfusionMatrix(num_classes=2)

    # Predicted as class 0: 1 sample, 1 correct (100%)
    # Predicted as class 1: 3 samples, 2 correct (67%)
    var preds_shape = List[Int]()
    preds_shape.append(4)  # 4 samples
    var preds = AnyTensor(preds_shape, DType.int32)
    var labels_shape = List[Int]()
    labels_shape.append(4)  # 4 samples
    var labels = AnyTensor(labels_shape, DType.int32)

    preds._data.bitcast[Int32]()[0] = 0  # ✓
    preds._data.bitcast[Int32]()[1] = 1  # ✗ (true=0)
    preds._data.bitcast[Int32]()[2] = 1  # ✓
    preds._data.bitcast[Int32]()[3] = 1  # ✓

    labels._data.bitcast[Int32]()[0] = 0
    labels._data.bitcast[Int32]()[1] = 0
    labels._data.bitcast[Int32]()[2] = 1
    labels._data.bitcast[Int32]()[3] = 1

    cm.update(preds, labels)

    var col_norm = cm.normalize(mode="column")

    # Column 0: [1, 0] -> [1.0, 0.0]
    # Column 1: [1, 2] -> [0.333..., 0.666...]

    assert_equal(
        col_norm._data.bitcast[Float64]()[0], 1.0, "Col 0, row 0 should be 1.0"
    )
    assert_equal(
        col_norm._data.bitcast[Float64]()[2], 0.0, "Col 0, row 1 should be 0.0"
    )

    var expected_col1_row0 = 1.0 / 3.0
    var expected_col1_row1 = 2.0 / 3.0
    var diff0 = abs(col_norm._data.bitcast[Float64]()[1] - expected_col1_row0)
    var diff1 = abs(col_norm._data.bitcast[Float64]()[3] - expected_col1_row1)

    assert_true(diff0 < 0.01, "Col 1, row 0 should be ~0.333")
    assert_true(diff1 < 0.01, "Col 1, row 1 should be ~0.667")

    print("  ✓ Column normalization test passed")


fn test_confusion_matrix_normalize_total() raises:
    """Test total normalization (percentages)."""
    print("Testing ConfusionMatrix total normalization...")

    var cm = ConfusionMatrix(num_classes=2)

    # 4 total samples
    var preds_shape = List[Int]()
    preds_shape.append(4)  # 4 samples
    var preds = AnyTensor(preds_shape, DType.int32)
    var labels_shape = List[Int]()
    labels_shape.append(4)  # 4 samples
    var labels = AnyTensor(labels_shape, DType.int32)

    preds._data.bitcast[Int32]()[0] = 0
    preds._data.bitcast[Int32]()[1] = 1
    preds._data.bitcast[Int32]()[2] = 1
    preds._data.bitcast[Int32]()[3] = 1

    labels._data.bitcast[Int32]()[0] = 0
    labels._data.bitcast[Int32]()[1] = 0
    labels._data.bitcast[Int32]()[2] = 1
    labels._data.bitcast[Int32]()[3] = 1

    cm.update(preds, labels)

    var total_norm = cm.normalize(mode="total")

    # Matrix: [1, 1; 0, 2]
    # Total: 4
    # Normalized: [0.25, 0.25; 0.0, 0.5]

    assert_equal(total_norm._data.bitcast[Float64]()[0], 0.25, "Should be 0.25")
    assert_equal(total_norm._data.bitcast[Float64]()[1], 0.25, "Should be 0.25")
    assert_equal(total_norm._data.bitcast[Float64]()[2], 0.0, "Should be 0.0")
    assert_equal(total_norm._data.bitcast[Float64]()[3], 0.5, "Should be 0.5")

    print("  ✓ Total normalization test passed")


fn test_confusion_matrix_precision() raises:
    """Test precision computation."""
    print("Testing ConfusionMatrix precision...")

    var cm = ConfusionMatrix(num_classes=3)

    # Class 0: predicted 2 times, 1 correct -> 50%
    # Class 1: predicted 2 times, 2 correct -> 100%
    # Class 2: predicted 1 time, 1 correct -> 100%
    var preds_shape = List[Int]()
    preds_shape.append(5)  # 5 samples
    var preds = AnyTensor(preds_shape, DType.int32)
    var labels_shape = List[Int]()
    labels_shape.append(5)  # 5 samples
    var labels = AnyTensor(labels_shape, DType.int32)

    preds._data.bitcast[Int32]()[0] = 0  # ✓
    preds._data.bitcast[Int32]()[1] = 0  # ✗ (true=1)
    preds._data.bitcast[Int32]()[2] = 1  # ✓
    preds._data.bitcast[Int32]()[3] = 1  # ✓
    preds._data.bitcast[Int32]()[4] = 2  # ✓

    labels._data.bitcast[Int32]()[0] = 0
    labels._data.bitcast[Int32]()[1] = 1
    labels._data.bitcast[Int32]()[2] = 1
    labels._data.bitcast[Int32]()[3] = 1
    labels._data.bitcast[Int32]()[4] = 2

    cm.update(preds, labels)

    var precision = cm.get_precision()

    assert_equal(
        precision._data.bitcast[Float64]()[0],
        0.5,
        "Precision class 0 should be 0.5",
    )
    assert_equal(
        precision._data.bitcast[Float64]()[1],
        1.0,
        "Precision class 1 should be 1.0",
    )
    assert_equal(
        precision._data.bitcast[Float64]()[2],
        1.0,
        "Precision class 2 should be 1.0",
    )

    print("  ✓ Precision test passed")


fn test_confusion_matrix_recall() raises:
    """Test recall computation."""
    print("Testing ConfusionMatrix recall...")

    var cm = ConfusionMatrix(num_classes=3)

    # Class 0: 1 sample, 1 correct -> 100%
    # Class 1: 3 samples, 2 correct -> 67%
    # Class 2: 1 sample, 1 correct -> 100%
    var preds_shape = List[Int]()
    preds_shape.append(5)  # 5 samples
    var preds = AnyTensor(preds_shape, DType.int32)
    var labels_shape = List[Int]()
    labels_shape.append(5)  # 5 samples
    var labels = AnyTensor(labels_shape, DType.int32)

    preds._data.bitcast[Int32]()[0] = 0  # ✓
    preds._data.bitcast[Int32]()[1] = 0  # ✗ (true=1)
    preds._data.bitcast[Int32]()[2] = 1  # ✓
    preds._data.bitcast[Int32]()[3] = 1  # ✓
    preds._data.bitcast[Int32]()[4] = 2  # ✓

    labels._data.bitcast[Int32]()[0] = 0
    labels._data.bitcast[Int32]()[1] = 1
    labels._data.bitcast[Int32]()[2] = 1
    labels._data.bitcast[Int32]()[3] = 1
    labels._data.bitcast[Int32]()[4] = 2

    cm.update(preds, labels)

    var recall = cm.get_recall()

    assert_equal(
        recall._data.bitcast[Float64]()[0], 1.0, "Recall class 0 should be 1.0"
    )

    var expected_recall_1 = 2.0 / 3.0
    var diff = abs(recall._data.bitcast[Float64]()[1] - expected_recall_1)
    assert_true(diff < 0.01, "Recall class 1 should be ~0.667")

    assert_equal(
        recall._data.bitcast[Float64]()[2], 1.0, "Recall class 2 should be 1.0"
    )

    print("  ✓ Recall test passed")


fn test_confusion_matrix_f1_score() raises:
    """Test F1-score computation."""
    print("Testing ConfusionMatrix F1-score...")

    var cm = ConfusionMatrix(num_classes=2)

    # Matrix: [[1, 1], [1, 1]]
    # Class 0: precision=0.5 (1/2), recall=0.5 (1/2) -> F1=0.5
    # Class 1: precision=0.5 (1/2), recall=0.5 (1/2) -> F1=0.5
    var preds_shape = List[Int]()
    preds_shape.append(4)  # 4 samples
    var preds = AnyTensor(preds_shape, DType.int32)
    var labels_shape = List[Int]()
    labels_shape.append(4)  # 4 samples
    var labels = AnyTensor(labels_shape, DType.int32)

    preds._data.bitcast[Int32]()[0] = 0  # ✓
    preds._data.bitcast[Int32]()[1] = 0  # ✗ (true=1)
    preds._data.bitcast[Int32]()[2] = 1  # ✓
    preds._data.bitcast[Int32]()[3] = 1  # ✗ (true=0)

    labels._data.bitcast[Int32]()[0] = 0
    labels._data.bitcast[Int32]()[1] = 1
    labels._data.bitcast[Int32]()[2] = 1
    labels._data.bitcast[Int32]()[3] = 0

    cm.update(preds, labels)

    var f1 = cm.get_f1_score()

    # F1 = 2 * (P * R) / (P + R)
    # Class 0: 2 * (0.5 * 0.5) / (0.5 + 0.5) = 0.5 / 1.0 = 0.5
    # Class 1: 2 * (0.5 * 0.5) / (0.5 + 0.5) = 0.5 / 1.0 = 0.5

    assert_equal(
        f1._data.bitcast[Float64]()[0], 0.5, "F1 class 0 should be 0.5"
    )
    assert_equal(
        f1._data.bitcast[Float64]()[1], 0.5, "F1 class 1 should be 0.5"
    )

    print("  ✓ F1-score test passed")


fn test_confusion_matrix_with_logits() raises:
    """Test confusion matrix with logits (not class indices)."""
    print("Testing ConfusionMatrix with logits...")

    var cm = ConfusionMatrix(num_classes=3)

    # Create logits [batch_size=4, num_classes=3]
    var logits_shape = List[Int]()
    logits_shape.append(4)
    logits_shape.append(3)
    var logits = AnyTensor(logits_shape, DType.float32)
    var labels_shape = List[Int]()
    labels_shape.append(4)  # 4 samples
    var labels = AnyTensor(labels_shape, DType.int32)

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
    var preds = AnyTensor(preds_shape, DType.int32)
    var labels_shape = List[Int]()
    labels_shape.append(2)  # 2 samples
    var labels = AnyTensor(labels_shape, DType.int32)
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
    var preds = AnyTensor(preds_shape, DType.int32)
    var labels = AnyTensor(preds_shape, DType.int32)
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
    var preds = AnyTensor(shape, DType.int32)
    var labels = AnyTensor(shape, DType.int32)

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
    """Run all test_confusion_matrix tests."""
    print("Running test_confusion_matrix tests...")

    test_confusion_matrix_basic()
    print("✓ test_confusion_matrix_basic")

    test_confusion_matrix_perfect()
    print("✓ test_confusion_matrix_perfect")

    test_confusion_matrix_normalize_row()
    print("✓ test_confusion_matrix_normalize_row")

    test_confusion_matrix_normalize_column()
    print("✓ test_confusion_matrix_normalize_column")

    test_confusion_matrix_normalize_total()
    print("✓ test_confusion_matrix_normalize_total")

    test_confusion_matrix_precision()
    print("✓ test_confusion_matrix_precision")

    test_confusion_matrix_recall()
    print("✓ test_confusion_matrix_recall")

    test_confusion_matrix_f1_score()
    print("✓ test_confusion_matrix_f1_score")

    test_confusion_matrix_with_logits()
    print("✓ test_confusion_matrix_with_logits")

    test_confusion_matrix_reset()
    print("✓ test_confusion_matrix_reset")

    test_confusion_matrix_empty()
    print("✓ test_confusion_matrix_empty")

    test_confusion_matrix_single_class()
    print("✓ test_confusion_matrix_single_class")

    test_confusion_matrix_misclassification()
    print("✓ test_confusion_matrix_misclassification")

    print("\nAll test_confusion_matrix tests passed!")
