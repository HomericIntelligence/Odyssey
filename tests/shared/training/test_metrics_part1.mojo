# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_metrics.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md

"""Tests for training metrics module - Part 1.

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
# ComponentTracker Tests (#283-287)
# ============================================================================


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


# ============================================================================
# LossTracker Tests (#283-287)
# ============================================================================


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


# ============================================================================
# Main Test Runner
# ============================================================================


fn main() raises:
    """Run ComponentTracker and LossTracker metrics tests."""
    print("=" * 60)
    print("Running Metrics Tests - Part 1 (ComponentTracker & LossTracker)")
    print("=" * 60)
    print()

    # ComponentTracker tests
    test_component_tracker_initialization()
    test_component_tracker_single_update()
    test_component_tracker_moving_average()
    test_component_tracker_statistics()
    test_component_tracker_reset()

    # LossTracker tests
    test_loss_tracker_single_component()
    test_loss_tracker_multi_component()
    test_loss_tracker_statistics()

    print()
    print("=" * 60)
    print("All Metrics Part 1 Tests Passed!")
    print("=" * 60)
