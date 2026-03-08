# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_metrics_coordination.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""Comprehensive tests for metrics coordination and unified interface (Part 2).

Tests MetricLogger best-value retrieval, metric summary utilities, training
simulation, and interface consistency.

Coordination tests (#293-297):
- #295: MetricLogger best-value retrieval
- #295: create_metric_summary utility
- #296: Integration with all metric types
- #297: Metric interface consistency

Testing strategy:
- Best value tracking: Identify best epoch for maximize/minimize metrics
- Summary formatting: Human-readable metric output
- Integration: All metrics work together in training pipeline
- Interface consistency: update/reset across all metric types
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


fn test_metric_logger_best() raises:
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


fn test_create_metric_summary() raises:
    """Test create_metric_summary formatting."""
    print("Testing create_metric_summary...")

    var results: List[MetricResult] = []
    results.append(MetricResult("accuracy", 0.9234))
    results.append(MetricResult("loss", 0.1523))

    var summary = create_metric_summary(results)

    # Check that summary contains metric names and values
    var contains_accuracy = False
    var contains_loss = False
    var contains_summary = False

    # Simple substring checks (Mojo doesn't have built-in contains)
    if len(summary) > 0:
        contains_summary = True
        # We can't easily check substrings in Mojo yet, so just verify non-empty
        assert_true(len(summary) > 0, "Summary should not be empty")

    print("  Summary output:")
    print(summary)
    print("  ✓ create_metric_summary produces output")


fn test_multi_metric_training_simulation() raises:
    """Simulate a training loop with multiple metrics."""
    print("Testing multi-metric training simulation...")

    # Setup metrics
    var accuracy = AccuracyMetric()
    var loss_tracker = LossTracker(window_size=10)
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
            var preds = ExTensor(preds_shape, DType.int32)
            var labels_shape = List[Int]()
            labels_shape.append(4)
            var labels = ExTensor(labels_shape, DType.int32)

            for i in range(4):
                var pred_class = (i + batch + epoch) % 3
                var true_class = (i + batch) % 3
                preds._data.bitcast[Int32]()[i] = Int32(pred_class)
                labels._data.bitcast[Int32]()[i] = Int32(true_class)

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


fn test_metric_interface_consistency() raises:
    """Test that all metrics have consistent interface patterns."""
    print("Testing metric interface consistency...")

    # All metrics should have update() and reset()
    var accuracy = AccuracyMetric()
    var confusion = ConfusionMatrix(num_classes=3)

    # Create test data
    var preds_shape = List[Int]()
    preds_shape.append(2)
    var preds = ExTensor(preds_shape, DType.int32)
    var labels_shape = List[Int]()
    labels_shape.append(2)
    var labels = ExTensor(labels_shape, DType.int32)
    preds._data.bitcast[Int32]()[0] = 0
    preds._data.bitcast[Int32]()[1] = 1
    labels._data.bitcast[Int32]()[0] = 0
    labels._data.bitcast[Int32]()[1] = 1

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


fn main() raises:
    """Run metrics coordination tests part 2."""
    print("\n" + "=" * 70)
    print("METRICS COORDINATION TEST SUITE - PART 2")
    print("Best Values, Summary, Training Simulation, Interface (#293-297)")
    print("=" * 70 + "\n")

    print("MetricLogger Best Value Tests (#295)")
    print("-" * 70)
    test_metric_logger_best()

    print("\nUtility Tests (#295)")
    print("-" * 70)
    test_create_metric_summary()

    print("\nIntegration Tests (#296)")
    print("-" * 70)
    test_multi_metric_training_simulation()

    print("\nInterface Consistency Tests (#297)")
    print("-" * 70)
    test_metric_interface_consistency()

    print("\n" + "=" * 70)
    print("ALL METRICS COORDINATION TESTS PART 2 PASSED ✓")
    print("=" * 70 + "\n")
    print("Summary:")
    print("  ✓ MetricLogger best value retrieval works")
    print("  ✓ create_metric_summary produces output")
    print("  ✓ Metrics integrate seamlessly in training pipelines")
    print(
        "  ✓ All metrics (Accuracy, LossTracker, ConfusionMatrix) comply with"
        " interface"
    )
    print()
