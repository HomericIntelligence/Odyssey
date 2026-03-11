# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_warmup_scheduler.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""Unit tests for Warmup Learning Rate Scheduler (Part 2).

Tests cover:
- Edge cases and error handling
- Property-based correctness tests

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


# ============================================================================
# Edge Cases and Error Handling
# ============================================================================


fn test_warmup_scheduler_zero_warmup_epochs() raises:
    """Test WarmupLR with warmup_epochs=0.

    Implementation handles this gracefully by returning base_lr.
    """
    var scheduler = WarmupLR(base_lr=1.0, warmup_epochs=0)

    # Should return base_lr immediately (defensive behavior)
    assert_almost_equal(scheduler.get_lr(0), 1.0)
    assert_almost_equal(scheduler.get_lr(10), 1.0)


fn test_warmup_scheduler_negative_warmup_epochs() raises:
    """Test WarmupLR with negative warmup_epochs.

    Implementation returns base_lr (defensive behavior).
    """
    var scheduler = WarmupLR(base_lr=1.0, warmup_epochs=-5)

    # Defensive behavior: return base_lr
    assert_almost_equal(scheduler.get_lr(0), 1.0)
    assert_almost_equal(scheduler.get_lr(10), 1.0)


fn test_warmup_scheduler_very_large_warmup() raises:
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


# ============================================================================
# Property-Based Tests
# ============================================================================


fn test_warmup_scheduler_property_monotonic_increase() raises:
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


fn test_warmup_scheduler_property_linear() raises:
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


fn test_warmup_scheduler_property_bounded() raises:
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


fn test_warmup_scheduler_property_starts_from_zero() raises:
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


# ============================================================================
# Test Main
# ============================================================================


fn main() raises:
    """Run WarmupLR scheduler tests (Part 2: edge cases, property-based tests).
    """
    print("Running edge cases...")
    test_warmup_scheduler_zero_warmup_epochs()
    test_warmup_scheduler_negative_warmup_epochs()
    test_warmup_scheduler_very_large_warmup()

    print("Running property-based tests...")
    test_warmup_scheduler_property_monotonic_increase()
    test_warmup_scheduler_property_linear()
    test_warmup_scheduler_property_bounded()
    test_warmup_scheduler_property_starts_from_zero()

    print("\nAll WarmupLR scheduler tests (Part 2) passed! ✓")
