# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_warmup_scheduler.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""Unit tests for Warmup Learning Rate Scheduler (Part 1).

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


# ============================================================================
# Warmup Scheduler Core Tests
# ============================================================================


fn test_warmup_scheduler_initialization() raises:
    """Test WarmupLR scheduler initialization with hyperparameters."""
    var scheduler = WarmupLR(base_lr=0.1, warmup_epochs=5)

    # Verify initial parameters
    assert_equal(scheduler.warmup_epochs, 5)
    assert_almost_equal(scheduler.base_lr, 0.1)


fn test_warmup_scheduler_linear_increase() raises:
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


fn test_warmup_scheduler_reaches_target() raises:
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


# ============================================================================
# Warmup Period Tests
# ============================================================================


fn test_warmup_scheduler_different_warmup_periods() raises:
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


fn test_warmup_scheduler_single_epoch_warmup() raises:
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


# ============================================================================
# Numerical Accuracy Tests
# ============================================================================


fn test_warmup_scheduler_matches_formula() raises:
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


fn test_warmup_scheduler_quarter_points() raises:
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


# ============================================================================
# Test Main
# ============================================================================


fn main() raises:
    """Run WarmupLR scheduler tests (Part 1: core, warmup period, numerical accuracy).
    """
    print("Running WarmupLR core tests...")
    test_warmup_scheduler_initialization()
    test_warmup_scheduler_linear_increase()
    test_warmup_scheduler_reaches_target()

    print("Running warmup period tests...")
    test_warmup_scheduler_different_warmup_periods()
    test_warmup_scheduler_single_epoch_warmup()

    print("Running numerical accuracy tests...")
    test_warmup_scheduler_matches_formula()
    test_warmup_scheduler_quarter_points()

    print("\nAll WarmupLR scheduler tests (Part 1) passed! ✓")
