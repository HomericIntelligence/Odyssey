"""Unit tests for Warmup Learning Rate Scheduler

Tests cover:
- Gradual LR increase from zero to target value
- Linear warmup strategy
- Warmup period configuration
- Mathematical correctness

All tests use the real WarmupLR implementation.
"""


from tests.shared.conftest import (
    assert_true,
    assert_equal,
    assert_almost_equal,
    assert_greater,
    assert_less,
    assert_greater_or_equal,
    assert_less_or_equal,
    TestFixtures,
)
from shared.training.schedulers import WarmupLR


def test_warmup_scheduler_initialization() raises:
    """Test WarmupLR scheduler initialization with hyperparameters."""
    var scheduler = WarmupLR(base_lr=0.1, warmup_epochs=5)

    # Verify initial parameters
    assert_equal(scheduler.warmup_epochs, 5)
    assert_almost_equal(scheduler.base_lr, 0.1)


def test_warmup_scheduler_linear_increase() raises:
    """Test WarmupLR increases LR linearly over warmup period.

    Formula: lr = base_lr * (epoch / warmup_epochs) for epoch < warmup_epochs
             lr = base_lr                            for epoch >= warmup_epochs

    The implementation starts from 0.0 and linearly increases to base_lr.
    """
    var scheduler = WarmupLR(base_lr=1.0, warmup_epochs=10)

    # Epoch 0: lr = 1.0 * (0/10) = 0.0
    var lr0 = scheduler.get_lr(epoch=0)
    assert_almost_equal(lr0, 0.0)

    # Epoch 1: lr = 1.0 * (1/10) = 0.1
    var lr1 = scheduler.get_lr(epoch=1)
    assert_almost_equal(lr1, 0.1)

    # Epoch 5: lr = 1.0 * (5/10) = 0.5
    var lr5 = scheduler.get_lr(epoch=5)
    assert_almost_equal(lr5, 0.5)

    # Epoch 10: lr = 1.0 (target reached)
    var lr10 = scheduler.get_lr(epoch=10)
    assert_almost_equal(lr10, 1.0)

    # Epoch 11+: lr remains at target
    var lr11 = scheduler.get_lr(epoch=11)
    assert_almost_equal(lr11, 1.0)


def test_warmup_scheduler_reaches_target() raises:
    """Test WarmupLR reaches and maintains target LR after warmup.

    After warmup_epochs:
    - LR equals base_lr
    - LR remains constant at base_lr.
    """
    var scheduler = WarmupLR(base_lr=0.1, warmup_epochs=5)

    # After warmup, should be at target
    var lr_after_warmup = scheduler.get_lr(epoch=5)
    assert_almost_equal(lr_after_warmup, 0.1)

    # Continue stepping - should remain at target
    var lr_later = scheduler.get_lr(epoch=20)
    assert_almost_equal(lr_later, 0.1)


def test_warmup_scheduler_different_warmup_periods() raises:
    """Test WarmupLR with different warmup_epochs values.

    warmup_epochs determines warmup speed:
    - Small warmup_epochs: Fast warmup
    - Large warmup_epochs: Slow warmup.
    """
    # Fast warmup (2 epochs)
    var scheduler1 = WarmupLR(base_lr=1.0, warmup_epochs=2)

    # Epoch 1: lr = 1.0 * (1/2) = 0.5
    var lr1_mid = scheduler1.get_lr(1)
    assert_almost_equal(lr1_mid, 0.5)

    # Epoch 2: lr = 1.0 (target)
    var lr1_end = scheduler1.get_lr(2)
    assert_almost_equal(lr1_end, 1.0)

    # Slow warmup (100 epochs)
    var scheduler2 = WarmupLR(base_lr=1.0, warmup_epochs=100)

    # Epoch 1: lr = 1.0 * (1/100) = 0.01
    var lr2_early = scheduler2.get_lr(1)
    assert_almost_equal(lr2_early, 0.01)


def test_warmup_scheduler_single_epoch_warmup() raises:
    """Test WarmupLR with warmup_epochs=1.

    Minimal warmup case:
    - Epoch 0: lr = 0.0
    - Epoch 1+: lr = base_lr.
    """
    var scheduler = WarmupLR(base_lr=1.0, warmup_epochs=1)

    # Epoch 0: lr = 0.0
    var lr0 = scheduler.get_lr(0)
    assert_almost_equal(lr0, 0.0)

    # Epoch 1: lr = 1.0
    var lr1 = scheduler.get_lr(1)
    assert_almost_equal(lr1, 1.0)


def test_warmup_scheduler_matches_formula() raises:
    """Test WarmupLR matches linear formula exactly.

    Formula: lr = base_lr * (epoch / warmup_epochs) for epoch < warmup_epochs

    Verifies mathematical correctness at multiple points.
    """
    var base_lr: Float64 = 0.5
    var warmup_epochs: Int = 20

    var scheduler = WarmupLR(base_lr=base_lr, warmup_epochs=warmup_epochs)

    # Test at several points during warmup
    for epoch in range(0, warmup_epochs + 1):
        var actual_lr = scheduler.get_lr(epoch)

        # Compute expected LR using formula
        var expected_lr: Float64
        if epoch >= warmup_epochs:
            expected_lr = base_lr
        else:
            expected_lr = base_lr * Float64(epoch) / Float64(warmup_epochs)

        assert_almost_equal(actual_lr, expected_lr, tolerance=1e-10)


def test_warmup_scheduler_quarter_points() raises:
    """Test WarmupLR at quarter-progress points for precision.

    Verifies linear progression at 0%, 25%, 50%, 75%, 100% of warmup.
    """
    var scheduler = WarmupLR(base_lr=1.0, warmup_epochs=100)

    # 0% (epoch 0): lr = 0.0
    assert_almost_equal(scheduler.get_lr(0), 0.0)

    # 25% (epoch 25): lr = 0.25
    assert_almost_equal(scheduler.get_lr(25), 0.25)

    # 50% (epoch 50): lr = 0.5
    assert_almost_equal(scheduler.get_lr(50), 0.5)

    # 75% (epoch 75): lr = 0.75
    assert_almost_equal(scheduler.get_lr(75), 0.75)

    # 100% (epoch 100): lr = 1.0
    assert_almost_equal(scheduler.get_lr(100), 1.0)


def test_warmup_scheduler_zero_warmup_epochs() raises:
    """Test WarmupLR with warmup_epochs=0.

    Implementation handles this gracefully by returning base_lr.
    """
    var scheduler = WarmupLR(base_lr=1.0, warmup_epochs=0)

    # Should return base_lr immediately (defensive behavior)
    assert_almost_equal(scheduler.get_lr(0), 1.0)
    assert_almost_equal(scheduler.get_lr(10), 1.0)


def test_warmup_scheduler_negative_warmup_epochs() raises:
    """Test WarmupLR with negative warmup_epochs.

    Implementation returns base_lr (defensive behavior).
    """
    var scheduler = WarmupLR(base_lr=1.0, warmup_epochs=-5)

    # Defensive behavior: return base_lr
    assert_almost_equal(scheduler.get_lr(0), 1.0)
    assert_almost_equal(scheduler.get_lr(10), 1.0)


def test_warmup_scheduler_very_large_warmup() raises:
    """Test WarmupLR with very large warmup_epochs.

    Should handle large periods without numerical issues.
    """
    var scheduler = WarmupLR(base_lr=1.0, warmup_epochs=10000)

    # Early epochs should have very small LR
    var lr_early = scheduler.get_lr(10)
    assert_almost_equal(lr_early, 0.001)

    # Middle epochs
    var lr_mid = scheduler.get_lr(5000)
    assert_almost_equal(lr_mid, 0.5)

    # End
    var lr_end = scheduler.get_lr(10000)
    assert_almost_equal(lr_end, 1.0)


def test_warmup_scheduler_property_monotonic_increase() raises:
    """Property: LR should monotonically increase during warmup.

    During warmup period (epoch < warmup_epochs), LR should never decrease.
    """
    var scheduler = WarmupLR(base_lr=1.0, warmup_epochs=20)

    var previous_lr = scheduler.get_lr(0)
    for epoch in range(1, 21):
        var current_lr = scheduler.get_lr(epoch)

        # LR should not decrease
        assert_greater_or_equal(current_lr, previous_lr)
        previous_lr = current_lr


def test_warmup_scheduler_property_linear() raises:
    """Property: LR increase should be perfectly linear.

    Equal epoch increments should produce equal LR increments.
    """
    var scheduler = WarmupLR(base_lr=1.0, warmup_epochs=10)

    # Collect LR values
    var lr0 = scheduler.get_lr(0)
    var lr1 = scheduler.get_lr(1)
    var lr2 = scheduler.get_lr(2)
    var lr3 = scheduler.get_lr(3)

    # Check linear spacing
    var increment1 = lr1 - lr0
    var increment2 = lr2 - lr1
    var increment3 = lr3 - lr2

    assert_almost_equal(increment1, increment2, tolerance=1e-10)
    assert_almost_equal(increment2, increment3, tolerance=1e-10)


def test_warmup_scheduler_property_bounded() raises:
    """Property: LR is always bounded by [0, base_lr].

    For all epochs: 0 <= LR <= base_lr.
    """
    var base_lr: Float64 = 0.5
    var scheduler = WarmupLR(base_lr=base_lr, warmup_epochs=20)

    for epoch in range(0, 30):
        var lr = scheduler.get_lr(epoch)

        # LR should be in bounds
        assert_greater_or_equal(lr, 0.0)
        assert_less_or_equal(lr, base_lr)


def test_warmup_scheduler_property_starts_from_zero() raises:
    """Property: WarmupLR always starts from 0.0 at epoch 0.

    Implementation starts from 0.0 (not configurable start_lr).
    """
    # Test with different base_lr values
    var scheduler1 = WarmupLR(base_lr=0.1, warmup_epochs=10)
    assert_almost_equal(scheduler1.get_lr(0), 0.0)

    var scheduler2 = WarmupLR(base_lr=1.0, warmup_epochs=100)
    assert_almost_equal(scheduler2.get_lr(0), 0.0)

    var scheduler3 = WarmupLR(base_lr=0.001, warmup_epochs=5)
    assert_almost_equal(scheduler3.get_lr(0), 0.0)


def main() raises:
    """Run all test_warmup_scheduler tests."""
    print("Running test_warmup_scheduler tests...")

    test_warmup_scheduler_initialization()
    print("✓ test_warmup_scheduler_initialization")

    test_warmup_scheduler_linear_increase()
    print("✓ test_warmup_scheduler_linear_increase")

    test_warmup_scheduler_reaches_target()
    print("✓ test_warmup_scheduler_reaches_target")

    test_warmup_scheduler_different_warmup_periods()
    print("✓ test_warmup_scheduler_different_warmup_periods")

    test_warmup_scheduler_single_epoch_warmup()
    print("✓ test_warmup_scheduler_single_epoch_warmup")

    test_warmup_scheduler_matches_formula()
    print("✓ test_warmup_scheduler_matches_formula")

    test_warmup_scheduler_quarter_points()
    print("✓ test_warmup_scheduler_quarter_points")

    test_warmup_scheduler_zero_warmup_epochs()
    print("✓ test_warmup_scheduler_zero_warmup_epochs")

    test_warmup_scheduler_negative_warmup_epochs()
    print("✓ test_warmup_scheduler_negative_warmup_epochs")

    test_warmup_scheduler_very_large_warmup()
    print("✓ test_warmup_scheduler_very_large_warmup")

    test_warmup_scheduler_property_monotonic_increase()
    print("✓ test_warmup_scheduler_property_monotonic_increase")

    test_warmup_scheduler_property_linear()
    print("✓ test_warmup_scheduler_property_linear")

    test_warmup_scheduler_property_bounded()
    print("✓ test_warmup_scheduler_property_bounded")

    test_warmup_scheduler_property_starts_from_zero()
    print("✓ test_warmup_scheduler_property_starts_from_zero")

    print("\nAll test_warmup_scheduler tests passed!")
