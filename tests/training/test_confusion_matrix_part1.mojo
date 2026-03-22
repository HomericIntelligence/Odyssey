"""Tests for confusion matrix metrics (Part 1 of 2).

Basic functionality, normalization, and derived metrics tests for ConfusionMatrix.

Test coverage:
- #289: Confusion matrix tests (part 1)

Testing strategy:
- Correctness: Verify matrix values and derived metrics
- Normalization: Test all modes (row, column, total, none)
- Multi-class: Test with various class distributions

ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
high test load. Split from test_confusion_matrix.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md

Note: Split from monolithic test file due to Mojo 0.26.1 heap corruption
bug that occurs after ~15 cumulative tests. See Issue #2942.
"""

from testing import assert_true, assert_false, assert_equal, assert_almost_equal
from shared.core.extensor import AnyTensor
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


fn main() raises:
    """Run confusion matrix tests (Part 1): basic, normalization, and derived metrics.
    """
    print("\n" + "=" * 70)
    print("CONFUSION MATRIX TEST SUITE - PART 1")
    print("=" * 70 + "\n")

    print("Basic Functionality Tests (#289)")
    print("-" * 70)
    test_confusion_matrix_basic()
    test_confusion_matrix_perfect()

    print("\nNormalization Tests (#289)")
    print("-" * 70)
    test_confusion_matrix_normalize_row()
    test_confusion_matrix_normalize_column()
    test_confusion_matrix_normalize_total()

    print("\nDerived Metrics Tests (#289)")
    print("-" * 70)
    test_confusion_matrix_precision()
    test_confusion_matrix_recall()
    test_confusion_matrix_f1_score()

    print("\n" + "=" * 70)
    print("ALL CONFUSION MATRIX PART 1 TESTS PASSED ✓")
    print("=" * 70 + "\n")
