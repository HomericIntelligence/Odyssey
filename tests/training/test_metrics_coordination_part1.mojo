# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_metrics_coordination.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""Comprehensive tests for metrics coordination and unified interface (Part 1).

Tests the Metric trait, MetricCollection, MetricLogger, and integration
of all metrics through consistent API.

Coordination tests (#293-297):
- #294: Unified metric interface validation
- #295: MetricCollection utilities
- #296: MetricLogger for history tracking
- #297: Integration with all metric types

Testing strategy:
- Interface compliance: All metrics implement Metric trait
- Collection management: Add/update/reset multiple metrics
- History logging: Track metrics across epochs
- Integration: All metrics work together in training pipeline
"""

from testing import assert_true, assert_false, assert_equal, assert_almost_equal
from shared.core import ExTensor
from shared.training.metrics import (
    Metric,
    MetricResult,
    MetricCollection,
    MetricLogger,
    create_metric_summary,
    AccuracyMetric,
    LossTracker,
    ConfusionMatrix,
)


fn test_metric_result_scalar() raises:
    """Test MetricResult with scalar values."""
    print("Testing MetricResult scalar...")

    var result = MetricResult("accuracy", 0.9234)

    assert_equal(result.name, "accuracy", "Result name")
    assert_true(result.is_scalar, "Result is scalar")
    assert_equal(result.scalar_value, 0.9234, "Scalar value")

    var value = result.get_scalar()
    assert_equal(value, 0.9234, "Get scalar value")

    # Should raise when accessing as tensor
    var raised = False
    try:
        var _ = result.get_tensor()
    except:
        raised = True

    assert_true(raised, "Should raise when accessing scalar as tensor")

    print("  ✓ MetricResult scalar works correctly")


fn test_metric_result_tensor() raises:
    """Test MetricResult with tensor values."""
    print("Testing MetricResult tensor...")

    var tensor_shape = List[Int]()
    tensor_shape.append(3)
    var tensor = ExTensor(tensor_shape, DType.float32)
    tensor._data.bitcast[Float32]()[0] = 0.9
    tensor._data.bitcast[Float32]()[1] = 0.8
    tensor._data.bitcast[Float32]()[2] = 0.95

    var result = MetricResult("per_class_acc", tensor)

    assert_equal(result.name, "per_class_acc", "Result name")
    assert_false(result.is_scalar, "Result is not scalar")

    var value_tensor = result.get_tensor()
    assert_equal(value_tensor.numel(), 3, "Tensor size")
    assert_equal(
        value_tensor._data.bitcast[Float32]()[0], 0.9, "Tensor value 0"
    )

    # Should raise when accessing as scalar
    var raised = False
    try:
        var _ = result.get_scalar()
    except:
        raised = True

    assert_true(raised, "Should raise when accessing tensor as scalar")

    print("  ✓ MetricResult tensor works correctly")


fn test_metric_collection_basic() raises:
    """Test basic MetricCollection operations."""
    print("Testing MetricCollection basic...")

    var collection = MetricCollection()

    assert_equal(len(collection), 0, "Initial size is 0")
    assert_false(
        collection.contains("accuracy"), "Does not contain accuracy initially"
    )

    # Add metrics
    collection.add("accuracy", AccuracyMetric())
    assert_equal(len(collection), 1, "Size after add")
    assert_true(collection.contains("accuracy"), "Contains accuracy")

    collection.add("loss", LossTracker(window_size=100))
    assert_equal(len(collection), 2, "Size after second add")
    assert_true(collection.contains("loss"), "Contains loss")

    # Get names
    var names = collection.get_names()
    assert_equal(len(names), 2, "Names vector size")
    assert_equal(names[0], "accuracy", "First name")
    assert_equal(names[1], "loss", "Second name")

    print("  ✓ MetricCollection basic operations work")


fn test_metric_collection_duplicate_handling() raises:
    """Test MetricCollection handles duplicate names."""
    print("Testing MetricCollection duplicate handling...")

    var collection = MetricCollection()

    collection.add("accuracy", AccuracyMetric())
    assert_equal(len(collection), 1, "Size after first add")

    # Add with same name - should warn and replace
    collection.add("accuracy", AccuracyMetric())
    assert_equal(len(collection), 1, "Size should stay 1 after duplicate")

    print("  ✓ MetricCollection handles duplicates correctly")


fn test_accuracy_metric_interface_compliance() raises:
    """Test AccuracyMetric implements Metric trait correctly."""
    print("Testing AccuracyMetric interface compliance...")

    var metric = AccuracyMetric()

    # Create test data
    var preds_shape = List[Int]()
    preds_shape.append(4)
    var preds = ExTensor(preds_shape, DType.int32)
    var labels_shape = List[Int]()
    labels_shape.append(4)
    var labels = ExTensor(labels_shape, DType.int32)

    preds._data.bitcast[Int32]()[0] = 0  # ✓
    preds._data.bitcast[Int32]()[1] = 1  # ✓
    preds._data.bitcast[Int32]()[2] = 2  # ✗
    preds._data.bitcast[Int32]()[3] = 0  # ✗

    labels._data.bitcast[Int32]()[0] = 0
    labels._data.bitcast[Int32]()[1] = 1
    labels._data.bitcast[Int32]()[2] = 0
    labels._data.bitcast[Int32]()[3] = 1

    # Update through Metric interface
    metric.update(preds, labels)

    var acc = metric.compute()
    assert_equal(acc, 0.5, "Accuracy should be 0.5")

    # Reset through Metric interface
    metric.reset()
    var acc_after_reset = metric.compute()
    assert_equal(acc_after_reset, 0.0, "Accuracy should be 0.0 after reset")

    print("  ✓ AccuracyMetric implements Metric trait correctly")


fn test_confusion_matrix_integration() raises:
    """Test ConfusionMatrix works with metric coordination."""
    print("Testing ConfusionMatrix integration...")

    var cm = ConfusionMatrix(num_classes=3)

    # Create test data
    var preds_shape = List[Int]()
    preds_shape.append(5)
    var preds = ExTensor(preds_shape, DType.int32)
    var labels_shape = List[Int]()
    labels_shape.append(5)
    var labels = ExTensor(labels_shape, DType.int32)

    preds._data.bitcast[Int32]()[0] = 0
    preds._data.bitcast[Int32]()[1] = 1
    preds._data.bitcast[Int32]()[2] = 2
    preds._data.bitcast[Int32]()[3] = 1
    preds._data.bitcast[Int32]()[4] = 2

    labels._data.bitcast[Int32]()[0] = 0
    labels._data.bitcast[Int32]()[1] = 1
    labels._data.bitcast[Int32]()[2] = 2
    labels._data.bitcast[Int32]()[3] = 0
    labels._data.bitcast[Int32]()[4] = 1

    # Update and compute precision
    cm.update(preds, labels)
    var precision = cm.get_precision()

    # Verify results
    assert_equal(precision.numel(), 3, "Precision has 3 classes")

    # Reset
    cm.reset()
    var raw_after_reset = cm.normalize(mode="none")
    var sum = 0
    for i in range(9):
        sum += Int(raw_after_reset._data.bitcast[Float64]()[i])
    assert_equal(sum, 0, "Matrix should be empty after reset")

    print("  ✓ ConfusionMatrix integrates correctly")


fn test_metric_logger_basic() raises:
    """Test basic MetricLogger functionality."""
    print("Testing MetricLogger basic...")

    var logger = MetricLogger()

    # Log first epoch
    var epoch0_metrics: List[MetricResult] = []
    epoch0_metrics.append(MetricResult("accuracy", 0.7))
    epoch0_metrics.append(MetricResult("loss", 0.5))

    logger.log_epoch(0, epoch0_metrics)

    assert_equal(logger.num_epochs, 1, "Logged 1 epoch")
    assert_equal(logger.num_metrics, 2, "Tracked 2 metrics")

    # Log second epoch
    var epoch1_metrics: List[MetricResult] = []
    epoch1_metrics.append(MetricResult("accuracy", 0.8))
    epoch1_metrics.append(MetricResult("loss", 0.4))

    logger.log_epoch(1, epoch1_metrics)

    assert_equal(logger.num_epochs, 2, "Logged 2 epochs")

    print("  ✓ MetricLogger basic operations work")


fn test_metric_logger_history() raises:
    """Test MetricLogger history retrieval."""
    print("Testing MetricLogger history...")

    var logger = MetricLogger()

    # Log multiple epochs
    for i in range(5):
        var metrics: List[MetricResult] = []
        var acc = 0.5 + Float64(i) * 0.1  # 0.5, 0.6, 0.7, 0.8, 0.9
        var loss = 1.0 - Float64(i) * 0.1  # 1.0, 0.9, 0.8, 0.7, 0.6
        metrics.append(MetricResult("accuracy", acc))
        metrics.append(MetricResult("loss", loss))
        logger.log_epoch(i, metrics)

    # Get history
    var acc_history = logger.get_history("accuracy")
    assert_equal(len(acc_history), 5, "Accuracy history length")
    assert_equal(acc_history[0], 0.5, "Accuracy epoch 0")
    assert_equal(acc_history[4], 0.9, "Accuracy epoch 4")

    var loss_history = logger.get_history("loss")
    assert_equal(len(loss_history), 5, "Loss history length")
    assert_equal(loss_history[0], 1.0, "Loss epoch 0")
    assert_equal(loss_history[4], 0.6, "Loss epoch 4")

    # Get latest
    var latest_acc = logger.get_latest("accuracy")
    assert_equal(latest_acc, 0.9, "Latest accuracy")

    var latest_loss = logger.get_latest("loss")
    assert_equal(latest_loss, 0.6, "Latest loss")

    print("  ✓ MetricLogger history retrieval works")


fn main() raises:
    """Run metrics coordination tests part 1."""
    print("\n" + "=" * 70)
    print("METRICS COORDINATION TEST SUITE - PART 1")
    print("MetricResult, MetricCollection, MetricLogger (#293-297)")
    print("=" * 70 + "\n")

    print("MetricResult Tests (#294)")
    print("-" * 70)
    test_metric_result_scalar()
    test_metric_result_tensor()

    print("\nMetricCollection Tests (#294)")
    print("-" * 70)
    test_metric_collection_basic()
    test_metric_collection_duplicate_handling()

    print("\nMetric Interface Compliance (#294)")
    print("-" * 70)
    test_accuracy_metric_interface_compliance()
    test_confusion_matrix_integration()

    print("\nMetricLogger Tests (#295)")
    print("-" * 70)
    test_metric_logger_basic()
    test_metric_logger_history()

    print("\n" + "=" * 70)
    print("ALL METRICS COORDINATION TESTS PART 1 PASSED ✓")
    print("=" * 70 + "\n")
    print("Summary:")
    print("  ✓ MetricResult handles both scalar and tensor metrics")
    print("  ✓ MetricCollection manages multiple metrics efficiently")
    print("  ✓ AccuracyMetric implements Metric trait correctly")
    print("  ✓ ConfusionMatrix integrates with metric coordination")
    print("  ✓ MetricLogger tracks metric history across epochs")
    print()
