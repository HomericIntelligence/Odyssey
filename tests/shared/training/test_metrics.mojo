"""Tests for training metrics module

Covers ComponentTracker and LossTracker tests.

Test coverage:
- #283-287: Loss tracking (ComponentTracker, LossTracker)
"""


from tests.shared.conftest import (
    assert_true,
    assert_false,
    assert_equal,
    assert_almost_equal,
)
from shared.tensor.any_tensor import AnyTensor, zeros, ones, full
from shared.training.metrics import (
    LossTracker,
    Statistics,
    ComponentTracker,
    MetricResult,
    create_metric_summary,
    MetricLogger,
)
from collections import List


fn test_component_tracker_initialization() raises:
    """Test ComponentTracker initializes correctly."""
    print("Testing ComponentTracker initialization...")

    var tracker = ComponentTracker(window_size=10)

    # Check initial state
    assert_equal(tracker.window_size, 10, "Window size should be 10")
    assert_equal(tracker.count, 0, "Initial count should be 0")
    assert_equal(tracker.buffer_idx, 0, "Initial buffer index should be 0")
    assert_false(tracker.buffer_full, "Buffer should not be full initially")

    print("   ComponentTracker initialization test passed")


fn test_component_tracker_single_update() raises:
    """Test ComponentTracker with single value."""
    print("Testing ComponentTracker single update...")

    var tracker = ComponentTracker(window_size=5)
    tracker.update(1.5)

    # Check state after one update
    assert_equal(tracker.count, 1, "Count should be 1")
    assert_almost_equal(
        tracker.get_current(), 1.5, 1e-6, "Current value should be 1.5"
    )
    assert_almost_equal(
        tracker.get_average(), 1.5, 1e-6, "Average should be 1.5"
    )

    var stats = tracker.get_statistics()
    assert_almost_equal(stats.mean, 1.5, 1e-6, "Mean should be 1.5")
    assert_equal(stats.count, 1, "Stats count should be 1")

    print("   ComponentTracker single update test passed")


fn test_component_tracker_moving_average() raises:
    """Test ComponentTracker moving average computation."""
    print("Testing ComponentTracker moving average...")

    var tracker = ComponentTracker(window_size=3)

    # Add 5 values, window should only keep last 3
    tracker.update(1.0)
    tracker.update(2.0)
    tracker.update(3.0)
    tracker.update(4.0)
    tracker.update(5.0)

    # Average of last 3: (3.0 + 4.0 + 5.0) / 3 = 4.0
    assert_almost_equal(
        tracker.get_average(), 4.0, 1e-6, "Moving average should be 4.0"
    )
    assert_almost_equal(
        tracker.get_current(), 5.0, 1e-6, "Current should be 5.0"
    )
    assert_equal(tracker.count, 5, "Total count should be 5")

    print("   ComponentTracker moving average test passed")


fn test_component_tracker_statistics() raises:
    """Test ComponentTracker statistics computation."""
    print("Testing ComponentTracker statistics...")

    var tracker = ComponentTracker(window_size=10)

    # Add known values
    tracker.update(1.0)
    tracker.update(2.0)
    tracker.update(3.0)
    tracker.update(4.0)
    tracker.update(5.0)

    var stats = tracker.get_statistics()

    # Mean should be 3.0
    assert_almost_equal(stats.mean, 3.0, 1e-6, "Mean should be 3.0")

    # Min/max
    assert_almost_equal(stats.min, 1.0, 1e-6, "Min should be 1.0")
    assert_almost_equal(stats.max, 5.0, 1e-6, "Max should be 5.0")

    # Count
    assert_equal(stats.count, 5, "Count should be 5")

    print("   ComponentTracker statistics test passed")


fn test_component_tracker_reset() raises:
    """Test ComponentTracker reset."""
    print("Testing ComponentTracker reset...")

    var tracker = ComponentTracker(window_size=5)

    tracker.update(1.0)
    tracker.update(2.0)
    tracker.update(3.0)

    # Reset
    tracker.reset()

    # Check all values are reset
    assert_equal(tracker.count, 0, "Count should be 0 after reset")
    assert_equal(tracker.buffer_idx, 0, "Buffer index should be 0 after reset")
    assert_false(tracker.buffer_full, "Buffer should not be full after reset")

    print("   ComponentTracker reset test passed")


fn test_loss_tracker_single_component() raises:
    """Test LossTracker with single component."""
    print("Testing LossTracker single component...")

    var tracker = LossTracker(window_size=10)

    # Add losses
    tracker.update(0.5, component="train")
    tracker.update(0.4, component="train")
    tracker.update(0.3, component="train")

    # Check current
    var current = tracker.get_current(component="train")
    assert_almost_equal(current, 0.3, 1e-6, "Current should be 0.3")

    # Check average
    var avg = tracker.get_average(component="train")
    var expected_avg = Float32((0.5 + 0.4 + 0.3) / 3.0)
    assert_almost_equal(
        avg, expected_avg, Float32(1e-6), "Average should be correct"
    )

    print("   LossTracker single component test passed")


fn test_loss_tracker_multi_component() raises:
    """Test LossTracker with multiple components."""
    print("Testing LossTracker multi-component...")

    var tracker = LossTracker(window_size=5)

    # Add different component losses
    tracker.update(1.0, component="total")
    tracker.update(0.6, component="reconstruction")
    tracker.update(0.4, component="regularization")

    tracker.update(0.9, component="total")
    tracker.update(0.5, component="reconstruction")
    tracker.update(0.4, component="regularization")

    # Check each component
    var total_avg = tracker.get_average(component="total")
    var recon_avg = tracker.get_average(component="reconstruction")
    var reg_avg = tracker.get_average(component="regularization")

    assert_almost_equal(total_avg, 0.95, 1e-6, "Total average should be 0.95")
    assert_almost_equal(recon_avg, 0.55, 1e-6, "Recon average should be 0.55")
    assert_almost_equal(reg_avg, 0.4, 1e-6, "Reg average should be 0.4")

    # Check component list
    var components = tracker.list_components()
    assert_equal(len(components), 3, "Should have 3 components")

    print("   LossTracker multi-component test passed")


fn test_loss_tracker_statistics() raises:
    """Test LossTracker statistics."""
    print("Testing LossTracker statistics...")

    var tracker = LossTracker(window_size=10)

    # Add known values
    tracker.update(1.0, component="loss")
    tracker.update(2.0, component="loss")
    tracker.update(3.0, component="loss")

    var stats = tracker.get_statistics(component="loss")

    assert_almost_equal(stats.mean, 2.0, 1e-6, "Mean should be 2.0")
    assert_almost_equal(stats.min, 1.0, 1e-6, "Min should be 1.0")
    assert_almost_equal(stats.max, 3.0, 1e-6, "Max should be 3.0")
    assert_equal(stats.count, 3, "Count should be 3")

    print("   LossTracker statistics test passed")


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


fn main() raises:
    """Run all test_metrics tests."""
    print("Running test_metrics tests...")

    test_component_tracker_initialization()
    print("✓ test_component_tracker_initialization")

    test_component_tracker_single_update()
    print("✓ test_component_tracker_single_update")

    test_component_tracker_moving_average()
    print("✓ test_component_tracker_moving_average")

    test_component_tracker_statistics()
    print("✓ test_component_tracker_statistics")

    test_component_tracker_reset()
    print("✓ test_component_tracker_reset")

    test_loss_tracker_single_component()
    print("✓ test_loss_tracker_single_component")

    test_loss_tracker_multi_component()
    print("✓ test_loss_tracker_multi_component")

    test_loss_tracker_statistics()
    print("✓ test_loss_tracker_statistics")

    test_loss_tracker_reset_all()
    print("✓ test_loss_tracker_reset_all")

    test_metric_result_scalar()
    print("✓ test_metric_result_scalar")

    test_metric_result_tensor()
    print("✓ test_metric_result_tensor")

    test_create_metric_summary()
    print("✓ test_create_metric_summary")

    test_metric_logger_single_epoch()
    print("✓ test_metric_logger_single_epoch")

    test_metric_logger_multiple_epochs()
    print("✓ test_metric_logger_multiple_epochs")

    test_metric_logger_get_latest()
    print("✓ test_metric_logger_get_latest")

    test_metric_logger_get_best()
    print("✓ test_metric_logger_get_best")

    print("\nAll test_metrics tests passed!")
