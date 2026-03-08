# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_metrics.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md

"""Tests for training metrics module - Part 2.

Covers LossTracker reset, MetricResult, and MetricLogger tests.

Test coverage:
- #283-287: Loss tracking (reset)
- #293-297: Base metric coordination
"""

from tests.shared.conftest import (
    assert_true,
    assert_false,
    assert_equal,
    assert_almost_equal,
)
from shared.core import ExTensor, zeros, ones, full
from shared.training.metrics import (
    LossTracker,
    Statistics,
    ComponentTracker,
    MetricResult,
    create_metric_summary,
    MetricLogger,
)
from collections import List


# ============================================================================
# LossTracker Reset Tests (#283-287)
# ============================================================================


fn test_loss_tracker_reset_all() raises:
    """Test LossTracker reset all components."""
    print("Testing LossTracker reset all...")

    var tracker = LossTracker(window_size=5)

    tracker.update(1.0, component="a")
    tracker.update(2.0, component="b")

    # Reset all
    tracker.reset(component="")

    # Both should be reset (return 0.0 for non-existent data)
    var avg_a = tracker.get_average(component="a")
    var avg_b = tracker.get_average(component="b")

    # After reset, averages should be 0.0 (no data)
    assert_almost_equal(avg_a, 0.0, 1e-6, "Component a should be reset")
    assert_almost_equal(avg_b, 0.0, 1e-6, "Component b should be reset")

    print("   LossTracker reset all test passed")


# ============================================================================
# MetricResult Tests (#293-297)
# ============================================================================


fn test_metric_result_scalar() raises:
    """Test MetricResult with scalar value."""
    print("Testing MetricResult scalar...")

    var result = MetricResult(name="accuracy", value=0.95)

    assert_true(result.is_scalar, "Should be scalar")
    assert_equal(result.name, "accuracy", "Name should be 'accuracy'")

    var val = result.get_scalar()
    assert_almost_equal(val, 0.95, 1e-6, "Scalar value should be 0.95")

    print("   MetricResult scalar test passed")


fn test_metric_result_tensor() raises:
    """Test MetricResult with tensor value."""
    print("Testing MetricResult tensor...")

    var tensor_shape = List[Int]()
    tensor_shape.append(3)
    var tensor = ones(tensor_shape, DType.float32)
    var result = MetricResult(name="per_class_acc", value=tensor)

    assert_false(result.is_scalar, "Should not be scalar")
    assert_equal(result.name, "per_class_acc", "Name should be 'per_class_acc'")

    var retrieved = result.get_tensor()
    assert_equal(retrieved.numel(), 3, "Tensor should have 3 elements")

    print("   MetricResult tensor test passed")


fn test_create_metric_summary() raises:
    """Test create_metric_summary utility."""
    print("Testing create_metric_summary...")

    var results: List[MetricResult] = []
    results.append(MetricResult(name="accuracy", value=0.95))
    results.append(MetricResult(name="loss", value=0.25))

    var summary = create_metric_summary(results)

    # Summary should contain both metrics
    assert_true(len(summary) > 0, "Summary should not be empty")
    # Can't easily check exact string, but should contain metric names

    print("   create_metric_summary test passed")


# ============================================================================
# MetricLogger Tests (#293-297)
# ============================================================================


fn test_metric_logger_single_epoch() raises:
    """Test MetricLogger with single epoch."""
    print("Testing MetricLogger single epoch...")

    var logger = MetricLogger()

    var metrics: List[MetricResult] = []
    metrics.append(MetricResult(name="accuracy", value=0.90))
    metrics.append(MetricResult(name="loss", value=0.50))

    logger.log_epoch(epoch=1, metrics=metrics)

    # Check history
    var acc_history = logger.get_history(metric_name="accuracy")
    assert_equal(len(acc_history), 1, "Should have 1 epoch")
    assert_almost_equal(acc_history[0], 0.90, 1e-6, "Accuracy should be 0.90")

    var loss_history = logger.get_history(metric_name="loss")
    assert_equal(len(loss_history), 1, "Should have 1 epoch")
    assert_almost_equal(loss_history[0], 0.50, 1e-6, "Loss should be 0.50")

    print("   MetricLogger single epoch test passed")


fn test_metric_logger_multiple_epochs() raises:
    """Test MetricLogger with multiple epochs."""
    print("Testing MetricLogger multiple epochs...")

    var logger = MetricLogger()

    # Epoch 1
    var metrics1: List[MetricResult] = []
    metrics1.append(MetricResult(name="accuracy", value=0.80))
    logger.log_epoch(epoch=1, metrics=metrics1)

    # Epoch 2
    var metrics2: List[MetricResult] = []
    metrics2.append(MetricResult(name="accuracy", value=0.85))
    logger.log_epoch(epoch=2, metrics=metrics2)

    # Epoch 3
    var metrics3: List[MetricResult] = []
    metrics3.append(MetricResult(name="accuracy", value=0.90))
    logger.log_epoch(epoch=3, metrics=metrics3)

    var history = logger.get_history(metric_name="accuracy")
    assert_equal(len(history), 3, "Should have 3 epochs")
    assert_almost_equal(history[0], 0.80, 1e-6, "Epoch 1 accuracy")
    assert_almost_equal(history[1], 0.85, 1e-6, "Epoch 2 accuracy")
    assert_almost_equal(history[2], 0.90, 1e-6, "Epoch 3 accuracy")

    print("   MetricLogger multiple epochs test passed")


fn test_metric_logger_get_latest() raises:
    """Test MetricLogger get_latest."""
    print("Testing MetricLogger get_latest...")

    var logger = MetricLogger()

    # Add several epochs
    var metrics1: List[MetricResult] = []
    metrics1.append(MetricResult(name="accuracy", value=0.80))
    logger.log_epoch(epoch=1, metrics=metrics1)

    var metrics2: List[MetricResult] = []
    metrics2.append(MetricResult(name="accuracy", value=0.95))
    logger.log_epoch(epoch=2, metrics=metrics2)

    var latest = logger.get_latest(metric_name="accuracy")
    assert_almost_equal(latest, 0.95, 1e-6, "Latest should be 0.95")

    print("   MetricLogger get_latest test passed")


fn test_metric_logger_get_best() raises:
    """Test MetricLogger get_best."""
    print("Testing MetricLogger get_best...")

    var logger = MetricLogger()

    # Add epochs with varying accuracy
    var metrics1: List[MetricResult] = []
    metrics1.append(MetricResult(name="accuracy", value=0.80))
    logger.log_epoch(epoch=1, metrics=metrics1)

    var metrics2: List[MetricResult] = []
    metrics2.append(MetricResult(name="accuracy", value=0.95))
    logger.log_epoch(epoch=2, metrics=metrics2)

    var metrics3: List[MetricResult] = []
    metrics3.append(MetricResult(name="accuracy", value=0.85))
    logger.log_epoch(epoch=3, metrics=metrics3)

    # Best accuracy (maximize=True)
    var best = logger.get_best(metric_name="accuracy", maximize=True)
    assert_almost_equal(best, 0.95, 1e-6, "Best accuracy should be 0.95")

    print("   MetricLogger get_best test passed")


# ============================================================================
# Main Test Runner
# ============================================================================


fn main() raises:
    """Run MetricResult and MetricLogger tests."""
    print("=" * 60)
    print("Running Metrics Tests - Part 2 (MetricResult & MetricLogger)")
    print("=" * 60)
    print()

    # LossTracker reset test
    test_loss_tracker_reset_all()

    # MetricResult tests
    test_metric_result_scalar()
    test_metric_result_tensor()
    test_create_metric_summary()

    # MetricLogger tests
    test_metric_logger_single_epoch()
    test_metric_logger_multiple_epochs()
    test_metric_logger_get_latest()
    test_metric_logger_get_best()

    print()
    print("=" * 60)
    print("All Metrics Part 2 Tests Passed!")
    print("=" * 60)
