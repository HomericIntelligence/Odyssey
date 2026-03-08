# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_step_scheduler.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""Unit tests for Step Learning Rate Scheduler (Part 1).

Tests cover:
- Step decay at fixed intervals
- Learning rate reduction by gamma factor
- Edge cases (step size, gamma values)

All tests use the real StepLR implementation.
"""

from tests.shared.conftest import (
    assert_true,
    assert_equal,
    assert_almost_equal,
    assert_greater,
    assert_less_or_equal,
    TestFixtures,
)
from shared.training.schedulers import StepLR


# ============================================================================
# Step Scheduler Core Tests
# ============================================================================


fn test_step_scheduler_initialization() raises:
    """Test StepLR scheduler initialization with hyperparameters."""
    var scheduler = StepLR(base_lr=0.1, step_size=10, gamma=0.1)

    # Verify initial parameters
    assert_equal(scheduler.step_size, 10)
    assert_almost_equal(scheduler.gamma, 0.1)
    assert_almost_equal(scheduler.base_lr, 0.1)


fn test_step_scheduler_reduces_lr_at_step() raises:
    """Test StepLR reduces learning rate at specified step.

    At epoch = step_size, LR *= gamma
    LR remains constant between steps.
    """
    var scheduler = StepLR(base_lr=1.0, step_size=5, gamma=0.1)

    # Initial LR (epoch 0)
    var lr0 = scheduler.get_lr(epoch=0)
    assert_almost_equal(lr0, 1.0)

    # Steps 1-4: LR unchanged
    for epoch in range(1, 5):
        var lr = scheduler.get_lr(epoch=epoch)
        assert_almost_equal(lr, 1.0)

    # Step 5: LR reduced
    var lr5 = scheduler.get_lr(epoch=5)
    assert_almost_equal(lr5, 0.1)


fn test_step_scheduler_multiple_steps() raises:
    """Test StepLR continues reducing LR at each step interval.

    LR reduction continues every step_size epochs:
    - Epoch 0-4: LR = initial_lr
    - Epoch 5-9: LR = initial_lr * gamma
    - Epoch 10-14: LR = initial_lr * gamma^2
    """
    var scheduler = StepLR(base_lr=1.0, step_size=5, gamma=0.1)

    # Epoch 0: LR = 1.0
    var lr0 = scheduler.get_lr(epoch=0)
    assert_almost_equal(lr0, 1.0)

    # Epoch 5: LR = 0.1
    var lr5 = scheduler.get_lr(epoch=5)
    assert_almost_equal(lr5, 0.1)

    # Epoch 10: LR = 0.01
    var lr10 = scheduler.get_lr(epoch=10)
    assert_almost_equal(lr10, 0.01)


# ============================================================================
# Gamma Factor Tests
# ============================================================================


fn test_step_scheduler_different_gamma_values() raises:
    """Test StepLR with different gamma (reduction factor) values.

    gamma determines how much LR is reduced:
    - gamma=0.1: LR reduced to 10% of previous value
    - gamma=0.5: LR reduced to 50% of previous value
    - gamma=0.9: LR reduced to 90% of previous value.
    """
    # Test gamma = 0.5
    var scheduler1 = StepLR(base_lr=1.0, step_size=1, gamma=0.5)
    assert_almost_equal(scheduler1.get_lr(1), 0.5)
    assert_almost_equal(scheduler1.get_lr(2), 0.25)

    # Test gamma = 0.9 (smaller reduction)
    var scheduler2 = StepLR(base_lr=1.0, step_size=1, gamma=0.9)
    assert_almost_equal(scheduler2.get_lr(1), 0.9)
    assert_almost_equal(scheduler2.get_lr(2), 0.81)


fn test_step_scheduler_gamma_one() raises:
    """Test StepLR with gamma=1.0 (no reduction).

    gamma=1.0 should result in no LR change.
    """
    var scheduler = StepLR(base_lr=1.0, step_size=1, gamma=1.0)

    # LR should remain constant
    for epoch in range(0, 11):
        assert_almost_equal(scheduler.get_lr(epoch), 1.0)


# ============================================================================
# Step Size Tests
# ============================================================================


fn test_step_scheduler_different_step_sizes() raises:
    """Test StepLR with different step_size values.

    step_size determines frequency of LR reduction:
    - step_size=1: Reduce every epoch
    - step_size=10: Reduce every 10 epochs.
    """
    # Test step_size = 1 (reduce every epoch)
    var scheduler1 = StepLR(base_lr=1.0, step_size=1, gamma=0.5)
    assert_almost_equal(scheduler1.get_lr(0), 1.0)
    assert_almost_equal(scheduler1.get_lr(1), 0.5)
    assert_almost_equal(scheduler1.get_lr(2), 0.25)

    # Test step_size = 10 (reduce every 10 epochs)
    var scheduler2 = StepLR(base_lr=1.0, step_size=10, gamma=0.5)

    # Epochs 0-9: No change
    for epoch in range(0, 10):
        assert_almost_equal(scheduler2.get_lr(epoch), 1.0)

    # Epoch 10: Reduce
    assert_almost_equal(scheduler2.get_lr(10), 0.5)


# ============================================================================
# Edge Cases and Error Handling
# ============================================================================


fn test_step_scheduler_zero_step_size() raises:
    """Test StepLR with step_size=0 returns base_lr.

    The implementation handles this gracefully by returning base_lr.
    """
    var scheduler = StepLR(base_lr=1.0, step_size=0, gamma=0.1)

    # Should return base_lr (defensive behavior)
    assert_almost_equal(scheduler.get_lr(0), 1.0)
    assert_almost_equal(scheduler.get_lr(10), 1.0)


fn test_step_scheduler_negative_gamma() raises:
    """Test StepLR with negative gamma.

    While mathematically valid, negative gamma would make LR oscillate.
    Current implementation allows it (no validation).
    """
    var scheduler = StepLR(base_lr=1.0, step_size=1, gamma=-0.5)

    # Mathematically: 1.0 * (-0.5)^1 = -0.5
    var lr1 = scheduler.get_lr(1)
    assert_almost_equal(lr1, -0.5)


# ============================================================================
# Test Main
# ============================================================================


fn main() raises:
    """Run StepLR scheduler tests (Part 1)."""
    print("Running StepLR core tests...")
    test_step_scheduler_initialization()
    test_step_scheduler_reduces_lr_at_step()
    test_step_scheduler_multiple_steps()

    print("Running gamma factor tests...")
    test_step_scheduler_different_gamma_values()
    test_step_scheduler_gamma_one()

    print("Running step size tests...")
    test_step_scheduler_different_step_sizes()

    print("Running edge cases and error handling...")
    test_step_scheduler_zero_step_size()
    test_step_scheduler_negative_gamma()

    print("\nAll StepLR scheduler tests (Part 1) passed! ✓")
