"""Tests for loss tracking metrics.

Comprehensive test suite for LossTracker with Welford's algorithm,
moving averages, and multi-component tracking.

Test coverage:
- #284: Loss tracking tests

Testing strategy:
- Correctness: Verify statistics match expected values
- Numerical stability: Test Welford's algorithm with large sequences
- Moving average: Verify window behavior
- Multi-component: Test independent component tracking
"""

from std.testing import (
    assert_true,
    assert_false,
    assert_equal,
    assert_almost_equal,
)
from std.math import sqrt
from projectodyssey.training.metrics import LossTracker, Statistics


def test_loss_tracker_basic() raises:
    """Test basic loss tracking functionality."""
    print("Testing LossTracker basic...")

    var tracker = LossTracker(window_size=10)

    # Add some values
    tracker.update(1.0, component="total")
    tracker.update(2.0, component="total")
    tracker.update(3.0, component="total")

    var current = tracker.get_current(component="total")
    var avg = tracker.get_average(component="total")

    print("  Current loss:", current)
    print("  Average loss:", avg)

    assert_equal(current, 3.0, "Current should be last value")
    assert_equal(avg, 2.0, "Average of [1, 2, 3] should be 2.0")

    print("  ✓ LossTracker basic test passed")


def test_loss_tracker_moving_average() raises:
    """Test moving average with window size."""
    print("Testing LossTracker moving average...")

    var tracker = LossTracker(window_size=3)

    # Add values: 1, 2, 3, 4, 5
    # Window should keep last 3: [3, 4, 5]
    tracker.update(1.0, component="total")
    tracker.update(2.0, component="total")
    tracker.update(3.0, component="total")
    tracker.update(4.0, component="total")
    tracker.update(5.0, component="total")

    var avg = tracker.get_average(component="total")

    print("  Moving average (last 3 of [1,2,3,4,5]):", avg)

    # Should be average of [3, 4, 5] = 4.0
    assert_equal(avg, 4.0, "Moving average should be 4.0")

    print("  ✓ LossTracker moving average test passed")


def test_loss_tracker_statistics() raises:
    """Test statistical summary computation."""
    print("Testing LossTracker statistics...")

    var tracker = LossTracker(window_size=100)

    # Add values: 1, 2, 3, 4, 5
    # Mean = 3.0
    # Variance = ((1-3)² + (2-3)² + (3-3)² + (4-3)² + (5-3)²) / 5
    #          = (4 + 1 + 0 + 1 + 4) / 5 = 2.0
    # Std = sqrt(2.0) ≈ 1.414

    tracker.update(1.0, component="total")
    tracker.update(2.0, component="total")
    tracker.update(3.0, component="total")
    tracker.update(4.0, component="total")
    tracker.update(5.0, component="total")

    var stats = tracker.get_statistics(component="total")

    print("  Mean:", stats.mean)
    print("  Std: ", stats.std)
    print("  Min: ", stats.min)
    print("  Max: ", stats.max)
    print("  Count:", stats.count)

    assert_equal(stats.mean, 3.0, "Mean should be 3.0")
    assert_equal(stats.min, 1.0, "Min should be 1.0")
    assert_equal(stats.max, 5.0, "Max should be 5.0")
    assert_equal(stats.count, 5, "Count should be 5")

    # Check std is approximately sqrt(2.0) ≈ 1.414
    var expected_std = Float32(sqrt(2.0))
    var diff = abs(stats.std - expected_std)
    assert_true(diff < 0.01, "Std should be approximately 1.414")

    print("  ✓ LossTracker statistics test passed")


def test_loss_tracker_welford_stability() raises:
    """Test Welford's algorithm numerical stability with large sequences."""
    print("Testing LossTracker Welford's algorithm stability...")

    var tracker = LossTracker(window_size=1000)

    # Add 100 values around 1.0
    for i in range(100):
        var value = 1.0 + Float32(i) * 0.01  # 1.0, 1.01, 1.02, ..., 1.99
        tracker.update(value, component="total")

    var stats = tracker.get_statistics(component="total")

    # Mean should be around 1.495
    var expected_mean = Float32(1.0 + 99.0 * 0.01 / 2.0)

    print("  Mean:", stats.mean)
    print("  Expected mean:", expected_mean)

    var mean_diff = abs(stats.mean - expected_mean)
    assert_true(mean_diff < 0.05, "Mean should be approximately correct")

    # Std should be computed stably even with small differences
    assert_true(stats.std > 0.0, "Std should be positive")
    assert_true(stats.std < 1.0, "Std should be reasonable")

    print("  ✓ Welford's algorithm stability test passed")


def test_loss_tracker_multi_component() raises:
    """Test multi-component loss tracking."""
    print("Testing LossTracker multi-component...")

    var tracker = LossTracker(window_size=10)

    # Track different components
    tracker.update(1.0, component="total")
    tracker.update(0.5, component="reconstruction")
    tracker.update(0.5, component="regularization")

    tracker.update(2.0, component="total")
    tracker.update(1.0, component="reconstruction")
    tracker.update(1.0, component="regularization")

    var total_avg = tracker.get_average(component="total")
    var recon_avg = tracker.get_average(component="reconstruction")
    var reg_avg = tracker.get_average(component="regularization")

    print("  Total avg:           ", total_avg)
    print("  Reconstruction avg:  ", recon_avg)
    print("  Regularization avg:  ", reg_avg)

    assert_equal(total_avg, 1.5, "Total average should be 1.5")
    assert_equal(recon_avg, 0.75, "Reconstruction average should be 0.75")
    assert_equal(reg_avg, 0.75, "Regularization average should be 0.75")

    # Verify components list
    var components = tracker.list_components()
    assert_equal(len(components), 3, "Should have 3 components")

    print("  ✓ Multi-component test passed")


def test_loss_tracker_reset() raises:
    """Test resetting tracker statistics."""
    print("Testing LossTracker reset...")

    var tracker = LossTracker(window_size=10)

    # Add values
    tracker.update(5.0, component="total")
    tracker.update(10.0, component="total")

    var avg_before = tracker.get_average(component="total")
    assert_equal(avg_before, 7.5, "Average should be 7.5 before reset")

    # Reset
    tracker.reset(component="total")

    var avg_after = tracker.get_average(component="total")
    var stats_after = tracker.get_statistics(component="total")

    print("  Average after reset:", avg_after)
    print("  Count after reset:  ", stats_after.count)

    assert_equal(avg_after, 0.0, "Average should be 0 after reset")
    assert_equal(stats_after.count, 0, "Count should be 0 after reset")

    print("  ✓ Reset test passed")


def test_loss_tracker_reset_all() raises:
    """Test resetting all components."""
    print("Testing LossTracker reset all...")

    var tracker = LossTracker(window_size=10)

    # Add values to multiple components
    tracker.update(1.0, component="loss1")
    tracker.update(2.0, component="loss2")
    tracker.update(3.0, component="loss3")

    # Reset all
    tracker.reset(component="")

    var stats1 = tracker.get_statistics(component="loss1")
    var stats2 = tracker.get_statistics(component="loss2")
    var stats3 = tracker.get_statistics(component="loss3")

    assert_equal(stats1.count, 0, "Loss1 should be reset")
    assert_equal(stats2.count, 0, "Loss2 should be reset")
    assert_equal(stats3.count, 0, "Loss3 should be reset")

    print("  ✓ Reset all test passed")


def test_loss_tracker_empty() raises:
    """Test tracker with no data."""
    print("Testing LossTracker empty...")

    var tracker = LossTracker(window_size=10)

    var current = tracker.get_current(component="nonexistent")
    var avg = tracker.get_average(component="nonexistent")
    var stats = tracker.get_statistics(component="nonexistent")

    assert_equal(current, 0.0, "Current should be 0 for nonexistent component")
    assert_equal(avg, 0.0, "Average should be 0 for nonexistent component")
    assert_equal(stats.count, 0, "Count should be 0 for nonexistent component")

    print("  ✓ Empty tracker test passed")


def test_loss_tracker_min_max() raises:
    """Test min/max tracking."""
    print("Testing LossTracker min/max...")

    var tracker = LossTracker(window_size=10)

    # Add values with clear min and max
    tracker.update(5.0, component="total")
    tracker.update(1.0, component="total")  # Min
    tracker.update(3.0, component="total")
    tracker.update(10.0, component="total")  # Max
    tracker.update(7.0, component="total")

    var stats = tracker.get_statistics(component="total")

    print("  Min:", stats.min)
    print("  Max:", stats.max)

    assert_equal(stats.min, 1.0, "Min should be 1.0")
    assert_equal(stats.max, 10.0, "Max should be 10.0")

    print("  ✓ Min/max test passed")


def test_loss_tracker_single_value() raises:
    """Test tracker with single value (edge case for std)."""
    print("Testing LossTracker single value...")

    var tracker = LossTracker(window_size=10)

    tracker.update(5.0, component="total")

    var stats = tracker.get_statistics(component="total")

    print("  Mean:", stats.mean)
    print("  Std: ", stats.std)
    print("  Count:", stats.count)

    assert_equal(stats.mean, 5.0, "Mean should be 5.0")
    assert_equal(stats.std, 0.0, "Std should be 0 for single value")
    assert_equal(stats.count, 1, "Count should be 1")

    print("  ✓ Single value test passed")


def main() raises:
    """Run all loss tracker tests."""
    print("\n" + "=" * 70)
    print("LOSS TRACKER TEST SUITE")
    print("=" * 70 + "\n")

    print("Basic Functionality Tests (#284)")
    print("-" * 70)
    test_loss_tracker_basic()
    test_loss_tracker_moving_average()
    test_loss_tracker_statistics()

    print("\nNumerical Stability Tests (#284)")
    print("-" * 70)
    test_loss_tracker_welford_stability()

    print("\nMulti-Component Tests (#284)")
    print("-" * 70)
    test_loss_tracker_multi_component()

    print("\nReset and Edge Cases (#284)")
    print("-" * 70)
    test_loss_tracker_reset()
    test_loss_tracker_reset_all()
    test_loss_tracker_empty()
    test_loss_tracker_min_max()
    test_loss_tracker_single_value()

    print("\n" + "=" * 70)
    print("ALL LOSS TRACKER TESTS PASSED ✓")
    print("=" * 70 + "\n")
