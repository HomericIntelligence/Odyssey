# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_step_scheduler.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""Unit tests for Step Learning Rate Scheduler (Part 2).

Tests cover:
- Very small learning rate behavior
- Property-based tests (monotonic decrease)
- Mathematical formula accuracy

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
# Edge Cases (continued)
# ============================================================================


fn test_step_scheduler_very_small_lr() raises:
    """Test StepLR continues to reduce LR even when very small.

    LR can become arbitrarily small (no minimum threshold).
    """
    var scheduler = StepLR(base_lr=1.0, step_size=1, gamma=0.1)

    # After 10 steps: LR = 1.0 * 0.1^10 = 1e-10
    var lr10 = scheduler.get_lr(10)
    assert_almost_equal(lr10, 1e-10, tolerance=1e-15)


# ============================================================================
# Property-Based Tests
# ============================================================================


fn test_step_scheduler_property_monotonic_decrease() raises:
    """Property: Learning rate should never increase (for 0 < gamma < 1).

    StepLR should only decrease or maintain LR, never increase it.
    """
    var scheduler = StepLR(base_lr=1.0, step_size=5, gamma=0.5)

    var previous_lr = scheduler.get_lr(0)
    for epoch in range(1, 51):
        var current_lr = scheduler.get_lr(epoch)

        # LR should not increase
        assert_less_or_equal(current_lr, previous_lr)
        previous_lr = current_lr


# ============================================================================
# Formula Accuracy Tests
# ============================================================================


fn test_step_scheduler_formula_accuracy() raises:
    """Test StepLR matches the mathematical formula exactly.

    Formula: lr = base_lr * gamma^(epoch // step_size).
    """
    var scheduler = StepLR(base_lr=0.1, step_size=30, gamma=0.1)

    # Epoch 0: gamma^0 = 1.0
    assert_almost_equal(scheduler.get_lr(0), 0.1)

    # Epoch 29: gamma^0 = 1.0
    assert_almost_equal(scheduler.get_lr(29), 0.1)

    # Epoch 30: gamma^1 = 0.1
    assert_almost_equal(scheduler.get_lr(30), 0.01)

    # Epoch 60: gamma^2 = 0.01
    assert_almost_equal(scheduler.get_lr(60), 0.001)


# ============================================================================
# Test Main
# ============================================================================


fn main() raises:
    """Run StepLR scheduler tests (Part 2)."""
    print("Running edge cases (continued)...")
    test_step_scheduler_very_small_lr()

    print("Running property-based tests...")
    test_step_scheduler_property_monotonic_decrease()

    print("Running formula accuracy tests...")
    test_step_scheduler_formula_accuracy()

    print("\nAll StepLR scheduler tests (Part 2) passed! ✓")
