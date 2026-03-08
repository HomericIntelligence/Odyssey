# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_schedulers.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""Scheduler edge case and integration tests - Part 3 of 3 (split per ADR-009).

Tests cover:
- CosineAnnealingLR: Edge cases and formula accuracy
- ReduceLROnPlateau: Edge cases and realistic training scenarios

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
# CosineAnnealingLR Edge Case Tests
# ============================================================================


fn test_cosine_annealing_zero_t_max() raises:
    """Test CosineAnnealingLR with T_max=0 (edge case).

    T_max <= 0 should return base_lr.
    """
    var scheduler = CosineAnnealingLR(base_lr=1.0, T_max=0, eta_min=0.0)

    var lr = scheduler.get_lr(0)
    assert_almost_equal(lr, 1.0)


fn test_cosine_annealing_formula_accuracy() raises:
    """Test CosineAnnealingLR matches mathematical formula exactly.

    Formula: lr = eta_min + (base_lr - eta_min) * (1 + cos(π * epoch / T_max)) / 2
    """
    var scheduler = CosineAnnealingLR(base_lr=0.1, T_max=10, eta_min=0.01)

    # Test several epochs
    var lr0 = scheduler.get_lr(0)
    # cos(0) = 1, so (1 + 1) / 2 = 1
    # LR = 0.01 + 0.09 * 1 = 0.1
    assert_almost_equal(lr0, 0.1, tolerance=1e-6)


# ============================================================================
# ReduceLROnPlateau Edge Case Tests
# ============================================================================


fn test_reduce_lr_on_plateau_factor_one() raises:
    """Test ReduceLROnPlateau with factor=1.0 (no reduction).

    factor=1.0 means LR is multiplied by 1.0, so no change.
    """
    var scheduler = ReduceLROnPlateau(
        base_lr=1.0, mode="min", factor=1.0, patience=1
    )

    _ = scheduler.step(0.5)
    _ = scheduler.step(0.6)  # No improvement
    var lr = scheduler.step(0.7)  # Reduce
    assert_almost_equal(lr, 1.0)  # No change


fn test_reduce_lr_on_plateau_zero_patience() raises:
    """Test ReduceLROnPlateau with patience=0 (reduce every epoch without improvement).

    patience=0 means reduce immediately on first no-improvement epoch.
    """
    var scheduler = ReduceLROnPlateau(
        base_lr=1.0, mode="min", factor=0.5, patience=0
    )

    _ = scheduler.step(0.5)
    var lr2 = scheduler.step(0.6)  # No improvement, reduce immediately
    assert_almost_equal(lr2, 0.5)


fn test_reduce_lr_on_plateau_very_small_lr() raises:
    """Test ReduceLROnPlateau can reduce LR to very small values.

    Multiple reductions can make LR arbitrarily small.
    """
    var scheduler = ReduceLROnPlateau(
        base_lr=1.0, mode="min", factor=0.1, patience=0
    )

    _ = scheduler.step(0.5)
    _ = scheduler.step(0.6)  # LR = 0.1
    _ = scheduler.step(0.7)  # LR = 0.01
    _ = scheduler.step(0.8)  # LR = 0.001
    var lr = scheduler.get_lr(0)

    assert_almost_equal(lr, 0.001, tolerance=1e-6)


fn test_reduce_lr_on_plateau_get_lr_interface() raises:
    """Test ReduceLROnPlateau implements LRScheduler.get_lr() interface.

    get_lr() should return current_lr regardless of epoch/batch.
    """
    var scheduler = ReduceLROnPlateau(
        base_lr=1.0, mode="min", factor=0.5, patience=2
    )

    _ = scheduler.step(0.5)
    _ = scheduler.step(0.6)
    _ = scheduler.step(0.7)  # LR reduced to 0.5

    # get_lr() should return current LR
    var lr0 = scheduler.get_lr(0)
    var lr10 = scheduler.get_lr(10)
    var lr100 = scheduler.get_lr(100)

    assert_almost_equal(lr0, 0.5)
    assert_almost_equal(lr10, 0.5)
    assert_almost_equal(lr100, 0.5)


# ============================================================================
# Integration Tests
# ============================================================================


fn test_reduce_lr_on_plateau_realistic_training_scenario() raises:
    """Test ReduceLROnPlateau in realistic training scenario.

    Simulates validation loss over multiple epochs with improvement plateau.
    """
    var scheduler = ReduceLROnPlateau(
        base_lr=0.01, mode="min", factor=0.5, patience=3
    )

    # Simulate validation loss decreasing then plateauing
    # Note: "no improvement" means value >= best, not just similar
    var val_losses: List[Float64] = [
        0.5,  # Epoch 0: improvement (best=0.5)
        0.4,  # Epoch 1: improvement (best=0.4)
        0.35,  # Epoch 2: improvement (best=0.35)
        0.36,  # Epoch 3: no improvement (0.36 > 0.35), counter=1
        0.37,  # Epoch 4: no improvement, counter=2
        0.38,  # Epoch 5: no improvement, counter=3 >= patience(3), reduce to 0.005
        0.34,  # Epoch 6: improvement (0.34 < 0.35), counter=0
        0.35,  # Epoch 7: no improvement, counter=1
        0.36,  # Epoch 8: no improvement, counter=2
        0.37,  # Epoch 9: no improvement, counter=3 >= patience(3), reduce again
    ]

    for loss in val_losses:
        _ = scheduler.step(loss)

    # After 2 reductions: 0.01 * 0.5 * 0.5 = 0.0025
    assert_almost_equal(scheduler.get_lr(0), 0.0025, tolerance=1e-6)


# ============================================================================
# Test Main
# ============================================================================


fn main() raises:
    """Run scheduler edge case and integration tests (Part 3)."""
    print("Running scheduler edge case and integration tests (Part 3)...")
    test_cosine_annealing_zero_t_max()
    test_cosine_annealing_formula_accuracy()
    test_reduce_lr_on_plateau_factor_one()
    test_reduce_lr_on_plateau_zero_patience()
    test_reduce_lr_on_plateau_very_small_lr()
    test_reduce_lr_on_plateau_get_lr_interface()
    test_reduce_lr_on_plateau_realistic_training_scenario()

    print("\nAll scheduler tests (Part 3) passed! ✓")
