"""Tests for evaluation utilities in shared.training.metrics.

Comprehensive test suite for evaluate functions that consolidate duplicated
evaluation patterns from train.mojo files across the examples directory.

Test coverage:
- evaluate(): Generic model evaluation with predict() method
- evaluate_batched(): Batched evaluation with forward() method
- compute_accuracy_on_batch(): Single batch accuracy computation
"""

from tests.shared.conftest import (
    assert_true,
    assert_false,
    assert_equal,
    assert_almost_equal,
)
from shared.core.extensor import AnyTensor, zeros, ones, full
from shared.training.metrics import (
    evaluate_with_predict,
    evaluate_logits_batch,
    compute_accuracy_on_batch,
)
from collections import List


# ============================================================================
# Mock Model for Testing
# ============================================================================


struct MockPredictor:
    """Mock model for testing evaluate() function."""

    var predictions: AnyTensor

    fn __init__(out self, var predictions: AnyTensor):
        """Initialize with fixed predictions.

        Args:
            predictions: Tensor of predicted class indices.
        """
        self.predictions = predictions^

    fn predict(self, sample: AnyTensor) raises -> Int:
        """Mock predict method that uses stored predictions.

        Args:
            sample: Single sample (not used, returns from predictions list).

        Returns:
            Predicted class index.
        """
        # This is a simplified mock - in real usage, this would process the sample
        return Int(self.predictions._data.bitcast[Int32]()[0])


# ============================================================================
# compute_accuracy_on_batch Tests
# ============================================================================


fn test_compute_accuracy_on_batch_perfect() raises:
    """Test compute_accuracy_on_batch with perfect predictions."""
    print("Testing compute_accuracy_on_batch with perfect predictions...")

    # Create logits: batch_size=4, num_classes=3
    var logits_shape = List[Int]()
    logits_shape.append(4)
    logits_shape.append(3)
    var logits = zeros(logits_shape, DType.float32)
    var logits_data = logits._data.bitcast[Float32]()

    # Set logits so argmax matches labels
    # Sample 0: logits [10, 0, 0] -> argmax=0
    logits_data[0] = 10.0
    logits_data[1] = 0.0
    logits_data[2] = 0.0

    # Sample 1: logits [0, 10, 0] -> argmax=1
    logits_data[3] = 0.0
    logits_data[4] = 10.0
    logits_data[5] = 0.0

    # Sample 2: logits [0, 0, 10] -> argmax=2
    logits_data[6] = 0.0
    logits_data[7] = 0.0
    logits_data[8] = 10.0

    # Sample 3: logits [10, 0, 0] -> argmax=0
    logits_data[9] = 10.0
    logits_data[10] = 0.0
    logits_data[11] = 0.0

    # Labels: [0, 1, 2, 0]
    var labels_shape_temp = List[Int]()
    labels_shape_temp.append(4)
    var labels = zeros(labels_shape_temp, DType.int32)
    var labels_data = labels._data.bitcast[Int32]()
    labels_data[0] = 0
    labels_data[1] = 1
    labels_data[2] = 2
    labels_data[3] = 0

    var accuracy = compute_accuracy_on_batch(logits, labels)

    assert_almost_equal(
        Float64(accuracy),
        1.0,
        1e-6,
        "Perfect predictions should have 100% accuracy",
    )
    print("   Perfect accuracy test passed")


fn test_compute_accuracy_on_batch_partial() raises:
    """Test compute_accuracy_on_batch with partial correct predictions."""
    print("Testing compute_accuracy_on_batch with partial predictions...")

    # Create logits: batch_size=4, num_classes=3
    var logits_shape_temp = List[Int]()
    logits_shape_temp.append(4)
    logits_shape_temp.append(3)
    var logits = zeros(logits_shape_temp, DType.float32)
    var logits_data = logits._data.bitcast[Float32]()

    # Sample 0: logits [10, 0, 0] -> argmax=0, label=0 ✓
    logits_data[0] = 10.0
    logits_data[1] = 0.0
    logits_data[2] = 0.0

    # Sample 1: logits [10, 0, 0] -> argmax=0, label=1 ✗
    logits_data[3] = 10.0
    logits_data[4] = 0.0
    logits_data[5] = 0.0

    # Sample 2: logits [0, 0, 10] -> argmax=2, label=2 ✓
    logits_data[6] = 0.0
    logits_data[7] = 0.0
    logits_data[8] = 10.0

    # Sample 3: logits [10, 0, 0] -> argmax=0, label=0 (correct)
    logits_data[9] = 10.0
    logits_data[10] = 0.0
    logits_data[11] = 0.0

    # Labels: [0, 1, 2, 0]
    var labels_shape_temp = List[Int]()
    labels_shape_temp.append(4)
    var labels = zeros(labels_shape_temp, DType.int32)
    var labels_data = labels._data.bitcast[Int32]()
    labels_data[0] = 0
    labels_data[1] = 1
    labels_data[2] = 2
    labels_data[3] = 0

    var accuracy = compute_accuracy_on_batch(logits, labels)

    # Expected: 3/4 = 0.75
    assert_almost_equal(
        Float64(accuracy), 0.75, 1e-6, "Expected 75% accuracy (3/4 correct)"
    )
    print("   Partial accuracy test passed")


fn test_compute_accuracy_on_batch_with_indices() raises:
    """Test compute_accuracy_on_batch with class indices (not logits)."""
    print("Testing compute_accuracy_on_batch with class indices...")

    # Predictions as 1D class indices: [0, 1, 2, 0]
    var predictions_shape = List[Int]()
    predictions_shape.append(4)
    var predictions = zeros(predictions_shape, DType.int32)
    var pred_data = predictions._data.bitcast[Int32]()
    pred_data[0] = 0
    pred_data[1] = 1
    pred_data[2] = 2
    pred_data[3] = 0

    # Labels: [0, 1, 2, 0]
    var labels_shape_temp = List[Int]()
    labels_shape_temp.append(4)
    var labels = zeros(labels_shape_temp, DType.int32)
    var labels_data = labels._data.bitcast[Int32]()
    labels_data[0] = 0
    labels_data[1] = 1
    labels_data[2] = 2
    labels_data[3] = 0

    var accuracy = compute_accuracy_on_batch(predictions, labels)

    assert_almost_equal(
        Float64(accuracy), 1.0, 1e-6, "100% accuracy with matching indices"
    )
    print("   Class indices test passed")


fn test_compute_accuracy_on_batch_zero() raises:
    """Test compute_accuracy_on_batch with zero correct predictions."""
    print("Testing compute_accuracy_on_batch with zero correct predictions...")

    # Create logits where all predictions are wrong
    var logits_shape_temp = List[Int]()
    logits_shape_temp.append(4)
    logits_shape_temp.append(3)
    var logits = zeros(logits_shape_temp, DType.float32)
    var logits_data = logits._data.bitcast[Float32]()

    # Sample 0: logits [10, 0, 0] -> argmax=0, label=1
    logits_data[0] = 10.0
    logits_data[1] = 0.0
    logits_data[2] = 0.0

    # Sample 1: logits [10, 0, 0] -> argmax=0, label=2
    logits_data[3] = 10.0
    logits_data[4] = 0.0
    logits_data[5] = 0.0

    # Sample 2: logits [10, 0, 0] -> argmax=0, label=1
    logits_data[6] = 10.0
    logits_data[7] = 0.0
    logits_data[8] = 0.0

    # Sample 3: logits [10, 0, 0] -> argmax=0, label=2
    logits_data[9] = 10.0
    logits_data[10] = 0.0
    logits_data[11] = 0.0

    # Labels: [1, 2, 1, 2] (all different from predicted 0)
    var labels_shape_temp = List[Int]()
    labels_shape_temp.append(4)
    var labels = zeros(labels_shape_temp, DType.int32)
    var labels_data = labels._data.bitcast[Int32]()
    labels_data[0] = 1
    labels_data[1] = 2
    labels_data[2] = 1
    labels_data[3] = 2

    var accuracy = compute_accuracy_on_batch(logits, labels)

    assert_almost_equal(
        Float64(accuracy),
        0.0,
        1e-6,
        "Zero correct predictions should have 0% accuracy",
    )
    print("   Zero accuracy test passed")


fn test_compute_accuracy_on_batch_single_sample() raises:
    """Test compute_accuracy_on_batch with single sample."""
    print("Testing compute_accuracy_on_batch with single sample...")

    # Single sample with logits [10, 0, 0] -> argmax=0
    var logits_shape_single = List[Int]()
    logits_shape_single.append(1)
    logits_shape_single.append(3)
    var logits = zeros(logits_shape_single, DType.float32)
    var logits_data = logits._data.bitcast[Float32]()
    logits_data[0] = 10.0
    logits_data[1] = 0.0
    logits_data[2] = 0.0

    # Label: 0
    var labels_shape_single = List[Int]()
    labels_shape_single.append(1)
    var labels = zeros(labels_shape_single, DType.int32)
    labels._data.bitcast[Int32]()[0] = 0

    var accuracy = compute_accuracy_on_batch(logits, labels)

    assert_almost_equal(
        Float64(accuracy), 1.0, 1e-6, "Single correct prediction should be 100%"
    )
    print("   Single sample test passed")


fn test_evaluate_logits_batch_perfect() raises:
    """Test evaluate_logits_batch with perfect predictions."""
    print("Testing evaluate_logits_batch with perfect predictions...")

    # Create logits: batch_size=4, num_classes=3
    var logits_shape_temp = List[Int]()
    logits_shape_temp.append(4)
    logits_shape_temp.append(3)
    var logits = zeros(logits_shape_temp, DType.float32)
    var logits_data = logits._data.bitcast[Float32]()

    # Set logits so argmax matches labels
    # Sample 0: logits [10, 0, 0] -> argmax=0
    logits_data[0] = 10.0
    logits_data[1] = 0.0
    logits_data[2] = 0.0

    # Sample 1: logits [0, 10, 0] -> argmax=1
    logits_data[3] = 0.0
    logits_data[4] = 10.0
    logits_data[5] = 0.0

    # Sample 2: logits [0, 0, 10] -> argmax=2
    logits_data[6] = 0.0
    logits_data[7] = 0.0
    logits_data[8] = 10.0

    # Sample 3: logits [10, 0, 0] -> argmax=0
    logits_data[9] = 10.0
    logits_data[10] = 0.0
    logits_data[11] = 0.0

    # Labels: [0, 1, 2, 0]
    var labels_shape_temp = List[Int]()
    labels_shape_temp.append(4)
    var labels = zeros(labels_shape_temp, DType.int32)
    var labels_data = labels._data.bitcast[Int32]()
    labels_data[0] = 0
    labels_data[1] = 1
    labels_data[2] = 2
    labels_data[3] = 0

    var accuracy = evaluate_logits_batch(logits, labels)

    assert_almost_equal(
        Float64(accuracy),
        1.0,
        1e-6,
        "Perfect predictions should have 100% accuracy",
    )
    print("   evaluate_logits_batch perfect test passed")


fn test_evaluate_with_predict() raises:
    """Test evaluate_with_predict with predictions list."""
    print("Testing evaluate_with_predict...")

    # Predictions as list: [0, 1, 2, 0]
    var predictions = List[Int]()
    predictions.append(0)
    predictions.append(1)
    predictions.append(2)
    predictions.append(0)

    # Labels: [0, 1, 2, 0]
    var labels_shape_temp = List[Int]()
    labels_shape_temp.append(4)
    var labels = zeros(labels_shape_temp, DType.int32)
    var labels_data = labels._data.bitcast[Int32]()
    labels_data[0] = 0
    labels_data[1] = 1
    labels_data[2] = 2
    labels_data[3] = 0

    var accuracy = evaluate_with_predict(predictions, labels)

    assert_almost_equal(
        Float64(accuracy),
        1.0,
        1e-6,
        "Perfect predictions should have 100% accuracy",
    )
    print("   evaluate_with_predict test passed")


# ============================================================================
# Main test function
# ============================================================================


fn main() raises:
    """Run all evaluate tests."""
    print("\n" + "=" * 60)
    print("Testing Evaluation Utilities (Issue #2291)")
    print("=" * 60 + "\n")

    # Test compute_accuracy_on_batch
    test_compute_accuracy_on_batch_perfect()
    test_compute_accuracy_on_batch_partial()
    test_compute_accuracy_on_batch_with_indices()
    test_compute_accuracy_on_batch_zero()
    test_compute_accuracy_on_batch_single_sample()

    # Test evaluate_logits_batch
    test_evaluate_logits_batch_perfect()

    # Test evaluate_with_predict
    test_evaluate_with_predict()

    print("\n" + "=" * 60)
    print("All evaluation tests passed!")
    print("=" * 60 + "\n")
