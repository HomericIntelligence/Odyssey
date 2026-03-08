# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_schedulers.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""ReduceLROnPlateau tests - Part 2 of 3 (split per ADR-009).

Tests cover:
- ReduceLROnPlateau: Metric-based learning rate reduction (initialization and basic modes)

All tests use the real scheduler implementations.
"""

from tests.shared.conftest import (
    assert_true,
    assert_equal,
    assert_almost_equal,
    assert_greater,
    assert_less_or_equal,
    TestFixtures,
)
from shared.training.schedulers import CosineAnnealingLR, ReduceLROnPlateau
from shared.training.schedulers.lr_schedulers import MODE_MIN, MODE_MAX


# ============================================================================
# ReduceLROnPlateau Tests
# ============================================================================


fn test_reduce_lr_on_plateau_initialization_min_mode() raises:
    """Test ReduceLROnPlateau initialization in 'min' mode."""
    var scheduler = ReduceLROnPlateau(
        base_lr=0.1, mode="min", factor=0.1, patience=10
    )

    # Verify initial parameters
    assert_almost_equal(scheduler.base_lr, 0.1)
    assert_equal(scheduler.mode, MODE_MIN)
    assert_almost_equal(scheduler.factor, 0.1)
    assert_equal(scheduler.patience, 10)

    # Initial LR should be base_lr
    var lr = scheduler.get_lr(0)
    assert_almost_equal(lr, 0.1)


fn test_reduce_lr_on_plateau_initialization_max_mode() raises:
    """Test ReduceLROnPlateau initialization in 'max' mode."""
    var scheduler = ReduceLROnPlateau(
        base_lr=0.1, mode="max", factor=0.1, patience=5
    )

    assert_equal(scheduler.mode, MODE_MAX)
    assert_equal(scheduler.patience, 5)


fn test_reduce_lr_on_plateau_min_mode_improvement() raises:
    """Test ReduceLROnPlateau detects improvement in 'min' mode.

    In 'min' mode, improvement means metric decreased.
    """
    var scheduler = ReduceLROnPlateau(
        base_lr=1.0, mode="min", factor=0.5, patience=3
    )

    # Metric improves (decreases)
    var lr1 = scheduler.step(0.5)
    assert_almost_equal(lr1, 1.0)  # No reduction yet

    var lr2 = scheduler.step(0.4)
    assert_almost_equal(lr2, 1.0)  # Still no reduction

    # Both steps show improvement, counter should reset
    assert_equal(scheduler.epochs_without_improvement, 0)


fn test_reduce_lr_on_plateau_min_mode_no_improvement() raises:
    """Test ReduceLROnPlateau detects no improvement in 'min' mode.

    In 'min' mode, no improvement means metric increased or stayed same.
    """
    var scheduler = ReduceLROnPlateau(
        base_lr=1.0, mode="min", factor=0.5, patience=2
    )

    # Initial metric
    var lr1 = scheduler.step(0.5)
    assert_almost_equal(lr1, 1.0)

    # Metric gets worse (increases) - no improvement
    var lr2 = scheduler.step(0.6)
    assert_almost_equal(lr2, 1.0)  # Not reduced yet (patience=2)
    assert_equal(scheduler.epochs_without_improvement, 1)

    # Still no improvement
    var lr3 = scheduler.step(0.7)
    assert_almost_equal(lr3, 0.5)  # Reduced after 2 epochs without improvement
    assert_equal(scheduler.epochs_without_improvement, 0)  # Counter reset


fn test_reduce_lr_on_plateau_max_mode_improvement() raises:
    """Test ReduceLROnPlateau detects improvement in 'max' mode.

    In 'max' mode, improvement means metric increased.
    """
    var scheduler = ReduceLROnPlateau(
        base_lr=1.0, mode="max", factor=0.5, patience=2
    )

    # Metric improves (increases)
    var lr1 = scheduler.step(0.5)
    assert_almost_equal(lr1, 1.0)

    var lr2 = scheduler.step(0.6)
    assert_almost_equal(lr2, 1.0)  # Still no reduction

    # Both show improvement, counter should be 0
    assert_equal(scheduler.epochs_without_improvement, 0)


fn test_reduce_lr_on_plateau_max_mode_no_improvement() raises:
    """Test ReduceLROnPlateau detects no improvement in 'max' mode.

    In 'max' mode, no improvement means metric decreased or stayed same.
    """
    var scheduler = ReduceLROnPlateau(
        base_lr=1.0, mode="max", factor=0.5, patience=2
    )

    # Initial metric
    var lr1 = scheduler.step(0.5)
    assert_almost_equal(lr1, 1.0)

    # Metric gets worse (decreases) - no improvement
    var lr2 = scheduler.step(0.4)
    assert_almost_equal(lr2, 1.0)  # Not reduced yet
    assert_equal(scheduler.epochs_without_improvement, 1)

    # Still no improvement
    var lr3 = scheduler.step(0.3)
    assert_almost_equal(lr3, 0.5)  # Reduced
    assert_equal(scheduler.epochs_without_improvement, 0)


fn test_reduce_lr_on_plateau_multiple_reductions() raises:
    """Test ReduceLROnPlateau continues reducing LR on multiple plateaus."""
    var scheduler = ReduceLROnPlateau(
        base_lr=1.0, mode="min", factor=0.5, patience=2
    )

    # First reduction
    _ = scheduler.step(0.5)
    _ = scheduler.step(0.6)  # No improvement
    var lr3 = scheduler.step(0.7)  # Still no improvement - reduce
    assert_almost_equal(lr3, 0.5)

    # Second reduction (continue from reduced LR)
    _ = scheduler.step(0.8)  # No improvement
    var lr5 = scheduler.step(0.9)  # Still no improvement - reduce again
    assert_almost_equal(lr5, 0.25)


fn test_reduce_lr_on_plateau_improvement_resets_counter() raises:
    """Test that improvement resets the no-improvement counter.

    After reaching patience and reducing LR, an improvement should reset counter.
    """
    var scheduler = ReduceLROnPlateau(
        base_lr=1.0, mode="min", factor=0.5, patience=2
    )

    # Accumulate no-improvement epochs
    _ = scheduler.step(0.5)
    _ = scheduler.step(0.6)  # No improvement, counter = 1
    _ = scheduler.step(0.7)  # No improvement, counter = 2, LR reduced to 0.5

    # Now improve
    var lr_improved = scheduler.step(0.4)  # Improvement! Counter resets
    assert_almost_equal(lr_improved, 0.5)  # LR unchanged
    assert_equal(scheduler.epochs_without_improvement, 0)


# ============================================================================
# Test Main
# ============================================================================


fn main() raises:
    """Run ReduceLROnPlateau tests (Part 2)."""
    print("Running ReduceLROnPlateau tests (Part 2)...")
    test_reduce_lr_on_plateau_initialization_min_mode()
    test_reduce_lr_on_plateau_initialization_max_mode()
    test_reduce_lr_on_plateau_min_mode_improvement()
    test_reduce_lr_on_plateau_min_mode_no_improvement()
    test_reduce_lr_on_plateau_max_mode_improvement()
    test_reduce_lr_on_plateau_max_mode_no_improvement()
    test_reduce_lr_on_plateau_multiple_reductions()
    test_reduce_lr_on_plateau_improvement_resets_counter()

    print("\nAll scheduler tests (Part 2) passed! ✓")
