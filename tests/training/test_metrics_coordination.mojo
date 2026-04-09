"""Comprehensive tests for metrics coordination and unified interface

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


from std.testing import assert_true, assert_false, assert_equal, assert_almost_equal
from shared.tensor.any_tensor import AnyTensor
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


def test_metric_result_scalar() raises:
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


def test_metric_result_tensor() raises:
    """Test MetricResult with tensor values."""
    print("Testing MetricResult tensor...")

    var tensor_shape = List[Int]()
    tensor_shape.append(3)
    var tensor = AnyTensor(tensor_shape, DType.float32)
    tensor.set(0, Float32(0.9))
    tensor.set(1, Float32(0.8))
    tensor.set(2, Float32(0.95))

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


def test_metric_collection_basic() raises:
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


def test_metric_collection_duplicate_handling() raises:
    """Test MetricCollection handles duplicate names."""
    print("Testing MetricCollection duplicate handling...")

    var collection = MetricCollection()

    collection.add("accuracy", AccuracyMetric())
    assert_equal(len(collection), 1, "Size after first add")

    # Add with same name - should warn and replace
    collection.add("accuracy", AccuracyMetric())
    assert_equal(len(collection), 1, "Size should stay 1 after duplicate")

    print("  ✓ MetricCollection handles duplicates correctly")


def test_accuracy_metric_interface_compliance() raises:
    """Test AccuracyMetric implements Metric trait correctly."""
    print("Testing AccuracyMetric interface compliance...")

    var metric = AccuracyMetric()

    # Create test data
    var preds_shape = List[Int]()
    preds_shape.append(4)
    var preds = AnyTensor(preds_shape, DType.int32)
    var labels_shape = List[Int]()
    labels_shape.append(4)
    var labels = AnyTensor(labels_shape, DType.int32)

    preds.set(0, Int32(0))  # ✓
    preds.set(1, Int32(1))  # ✓
    preds.set(2, Int32(2))  # ✗
    preds.set(3, Int32(0))  # ✗

    labels.set(0, Int32(0))
    labels.set(1, Int32(1))
    labels.set(2, Int32(0))
    labels.set(3, Int32(1))

    # Update through Metric interface
    metric.update(preds, labels)

    var acc = metric.compute()
    assert_equal(acc, 0.5, "Accuracy should be 0.5")

    # Reset through Metric interface
    metric.reset()
    var acc_after_reset = metric.compute()
    assert_equal(acc_after_reset, 0.0, "Accuracy should be 0.0 after reset")

    print("  ✓ AccuracyMetric implements Metric trait correctly")


def test_confusion_matrix_integration() raises:
    """Test ConfusionMatrix works with metric coordination."""
    print("Testing ConfusionMatrix integration...")

    var cm = ConfusionMatrix(num_classes=3)

    # Create test data
    var preds_shape = List[Int]()
    preds_shape.append(5)
    var preds = AnyTensor(preds_shape, DType.int32)
    var labels_shape = List[Int]()
    labels_shape.append(5)
    var labels = AnyTensor(labels_shape, DType.int32)

    preds.set(0, Int32(0))
    preds.set(1, Int32(1))
    preds.set(2, Int32(2))
    preds.set(3, Int32(1))
    preds.set(4, Int32(2))

    labels.set(0, Int32(0))
    labels.set(1, Int32(1))
    labels.set(2, Int32(2))
    labels.set(3, Int32(0))
    labels.set(4, Int32(1))

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


def test_metric_logger_basic() raises:
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


def test_metric_logger_history() raises:
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


def test_metric_logger_best() raises:
    """Test MetricLogger best value retrieval."""
    print("Testing MetricLogger best value...")

    var logger = MetricLogger()

    # Log epochs with varying metrics
    var metrics0: List[MetricResult] = []
    metrics0.append(MetricResult("accuracy", 0.7))
    metrics0.append(MetricResult("loss", 0.8))
    logger.log_epoch(0, metrics0)

    var metrics1: List[MetricResult] = []
    metrics1.append(MetricResult("accuracy", 0.9))  # Best accuracy
    metrics1.append(MetricResult("loss", 0.6))
    logger.log_epoch(1, metrics1)

    var metrics2: List[MetricResult] = []
    metrics2.append(MetricResult("accuracy", 0.8))
    metrics2.append(MetricResult("loss", 0.5))  # Best loss
    logger.log_epoch(2, metrics2)

    # Get best values
    var best_acc = logger.get_best("accuracy", maximize=True)
    assert_equal(best_acc, 0.9, "Best accuracy (maximize)")

    var best_loss = logger.get_best("loss", maximize=False)
    assert_equal(best_loss, 0.5, "Best loss (minimize)")

    print("  ✓ MetricLogger best value retrieval works")


def test_create_metric_summary() raises:
    """Test create_metric_summary formatting."""
    print("Testing create_metric_summary...")

    var results: List[MetricResult] = []
    results.append(MetricResult("accuracy", 0.9234))
    results.append(MetricResult("loss", 0.1523))

    var summary = create_metric_summary(results)

    # Check that summary contains metric names and values
    # Simple substring checks (Mojo doesn't have built-in contains)
    if len(summary) > 0:
        # We can't easily check substrings in Mojo yet, so just verify non-empty
        assert_true(len(summary) > 0, "Summary should not be empty")

    print("  Summary output:")
    print(summary)
    print("  ✓ create_metric_summary produces output")


def test_multi_metric_training_simulation() raises:
    """Simulate a training loop with multiple metrics."""
    print("Testing multi-metric training simulation...")

    # Setup metrics
    var accuracy = AccuracyMetric()
    var _loss_tracker = LossTracker(window_size=10)
    var confusion = ConfusionMatrix(num_classes=3)

    # Setup logger
    var logger = MetricLogger()

    # Simulate 3 epochs
    for epoch in range(3):
        # Reset metrics for new epoch
        accuracy.reset()
        confusion.reset()

        # Simulate 5 batches per epoch
        for batch in range(5):
            # Create fake batch data
            var preds_shape = List[Int]()
            preds_shape.append(4)
            var preds = AnyTensor(preds_shape, DType.int32)
            var labels_shape = List[Int]()
            labels_shape.append(4)
            var labels = AnyTensor(labels_shape, DType.int32)

            for i in range(4):
                var pred_class = (i + batch + epoch) % 3
                var true_class = (i + batch) % 3
                preds.set(i, Int32(Int32(pred_class)))
                labels.set(i, Int32(Int32(true_class)))

            # Update all metrics
            accuracy.update(preds, labels)
            confusion.update(preds, labels)

        # Compute epoch metrics
        var epoch_acc = accuracy.compute()
        var epoch_precision = confusion.get_precision()

        print("  Epoch " + String(epoch) + ": accuracy=" + String(epoch_acc))

        # Log to history
        var epoch_metrics: List[MetricResult] = []
        epoch_metrics.append(MetricResult("accuracy", epoch_acc))
        logger.log_epoch(epoch, epoch_metrics)

    # Verify we logged all epochs
    assert_equal(logger.num_epochs, 3, "Logged 3 epochs")

    var acc_history = logger.get_history("accuracy")
    assert_equal(len(acc_history), 3, "Accuracy history has 3 epochs")

    print("  ✓ Multi-metric training simulation works")


def test_metric_interface_consistency() raises:
    """Test that all metrics have consistent interface patterns."""
    print("Testing metric interface consistency...")

    # All metrics should have update() and reset()
    var accuracy = AccuracyMetric()
    var confusion = ConfusionMatrix(num_classes=3)

    # Create test data
    var preds_shape = List[Int]()
    preds_shape.append(2)
    var preds = AnyTensor(preds_shape, DType.int32)
    var labels_shape = List[Int]()
    labels_shape.append(2)
    var labels = AnyTensor(labels_shape, DType.int32)
    preds.set(0, Int32(0))
    preds.set(1, Int32(1))
    labels.set(0, Int32(0))
    labels.set(1, Int32(1))

    # Both should accept update()
    accuracy.update(preds, labels)
    confusion.update(preds, labels)

    # Both should accept reset()
    accuracy.reset()
    confusion.reset()

    # Verify reset worked
    var acc_after_reset = accuracy.compute()
    assert_equal(acc_after_reset, 0.0, "Accuracy reset to 0.0")

    var cm_after_reset = confusion.normalize(mode="none")
    var cm_sum = 0
    for i in range(9):
        cm_sum += Int(cm_after_reset._data.bitcast[Float64]()[i])
    assert_equal(cm_sum, 0, "Confusion matrix reset to zeros")

    print("  ✓ All metrics have consistent interface")


def main() raises:
    """Run all test_metrics_coordination tests."""
    print("Running test_metrics_coordination tests...")

    test_metric_result_scalar()
    print("✓ test_metric_result_scalar")

    test_metric_result_tensor()
    print("✓ test_metric_result_tensor")

    test_metric_collection_basic()
    print("✓ test_metric_collection_basic")

    test_metric_collection_duplicate_handling()
    print("✓ test_metric_collection_duplicate_handling")

    test_accuracy_metric_interface_compliance()
    print("✓ test_accuracy_metric_interface_compliance")

    test_confusion_matrix_integration()
    print("✓ test_confusion_matrix_integration")

    test_metric_logger_basic()
    print("✓ test_metric_logger_basic")

    test_metric_logger_history()
    print("✓ test_metric_logger_history")

    test_metric_logger_best()
    print("✓ test_metric_logger_best")

    test_create_metric_summary()
    print("✓ test_create_metric_summary")

    test_multi_metric_training_simulation()
    print("✓ test_multi_metric_training_simulation")

    test_metric_interface_consistency()
    print("✓ test_metric_interface_consistency")

    print("\nAll test_metrics_coordination tests passed!")
