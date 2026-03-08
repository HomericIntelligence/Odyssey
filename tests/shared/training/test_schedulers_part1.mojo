# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_schedulers.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""CosineAnnealingLR tests - Part 1 of 3 (split per ADR-009).

Tests cover:
- CosineAnnealingLR: Smooth cosine decay (initialization and boundary tests)

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
# CosineAnnealingLR Tests
# ============================================================================


fn test_cosine_annealing_initialization() raises:
    """Test CosineAnnealingLR scheduler initialization."""
    var scheduler = CosineAnnealingLR(base_lr=0.1, T_max=100, eta_min=0.0)

    # Verify initial parameters
    assert_almost_equal(scheduler.base_lr, 0.1)
    assert_equal(scheduler.T_max, 100)
    assert_almost_equal(scheduler.eta_min, 0.0)


fn test_cosine_annealing_epoch_zero() raises:
    """Test CosineAnnealingLR at epoch 0 (maximum learning rate).

    At epoch 0, LR should equal base_lr.
    Formula: lr = eta_min + (base_lr - eta_min) * (1 + cos(π * 0 / T_max)) / 2
    = eta_min + (base_lr - eta_min) * (1 + 1) / 2
    = base_lr.
    """
    var scheduler = CosineAnnealingLR(base_lr=0.1, T_max=100, eta_min=0.0)

    var lr0 = scheduler.get_lr(epoch=0)
    assert_almost_equal(lr0, 0.1)


fn test_cosine_annealing_epoch_max() raises:
    """Test CosineAnnealingLR at epoch T_max (minimum learning rate).

    At epoch T_max, LR should equal eta_min.
    Formula: lr = eta_min + (base_lr - eta_min) * (1 + cos(π * 1)) / 2
    = eta_min + (base_lr - eta_min) * (1 - 1) / 2
    = eta_min.
    """
    var scheduler = CosineAnnealingLR(base_lr=0.1, T_max=100, eta_min=0.01)

    var lr_max = scheduler.get_lr(epoch=100)
    assert_almost_equal(lr_max, 0.01, tolerance=1e-6)


fn test_cosine_annealing_midpoint() raises:
    """Test CosineAnnealingLR at midpoint epoch.

    At epoch = T_max / 2, cosine factor should be 0.
    LR = eta_min + (base_lr - eta_min) * 0 / 2 = eta_min.
    """
    var scheduler = CosineAnnealingLR(base_lr=1.0, T_max=100, eta_min=0.0)

    var lr_mid = scheduler.get_lr(epoch=50)
    # At midpoint: cos(π/2) = 0, so (1 + 0) / 2 = 0.5
    # LR = 0 + 1.0 * 0.5 = 0.5
    assert_almost_equal(lr_mid, 0.5, tolerance=1e-6)


fn test_cosine_annealing_smooth_decay() raises:
    """Test CosineAnnealingLR decays smoothly over epochs.

    Learning rate should decrease monotonically (for eta_min < base_lr).
    """
    var scheduler = CosineAnnealingLR(base_lr=1.0, T_max=100, eta_min=0.0)

    var previous_lr = scheduler.get_lr(0)
    for epoch in range(1, 101):
        var current_lr = scheduler.get_lr(epoch)

        # LR should decrease or stay the same
        assert_less_or_equal(current_lr, previous_lr)
        previous_lr = current_lr


fn test_cosine_annealing_with_eta_min() raises:
    """Test CosineAnnealingLR respects eta_min floor.

    LR should never go below eta_min.
    """
    var scheduler = CosineAnnealingLR(base_lr=1.0, T_max=100, eta_min=0.1)

    for epoch in range(0, 101):
        var lr = scheduler.get_lr(epoch)

        # LR should not go below eta_min
        assert_greater(lr, 0.09)  # Small tolerance for floating point


fn test_cosine_annealing_different_t_max() raises:
    """Test CosineAnnealingLR with different T_max values.

    Larger T_max should result in slower decay.
    """
    var scheduler1 = CosineAnnealingLR(base_lr=1.0, T_max=50, eta_min=0.0)
    var scheduler2 = CosineAnnealingLR(base_lr=1.0, T_max=200, eta_min=0.0)

    # At epoch 50
    var lr1_50 = scheduler1.get_lr(50)
    var lr2_50 = scheduler2.get_lr(50)

    # scheduler1 completes its cycle at epoch 50, so LR is near eta_min
    # scheduler2 is still decaying at epoch 50, so LR is higher
    assert_less_or_equal(lr1_50, lr2_50)


fn test_cosine_annealing_beyond_t_max() raises:
    """Test CosineAnnealingLR beyond T_max (clamped to T_max).

    Epochs beyond T_max should clamp to T_max.
    """
    var scheduler = CosineAnnealingLR(base_lr=1.0, T_max=100, eta_min=0.0)

    var lr_100 = scheduler.get_lr(100)
    var lr_200 = scheduler.get_lr(200)

    # Both should give the same result (clamped at T_max)
    assert_almost_equal(lr_100, lr_200)


# ============================================================================
# Test Main
# ============================================================================


fn main() raises:
    """Run CosineAnnealingLR tests (Part 1)."""
    print("Running CosineAnnealingLR tests (Part 1)...")
    test_cosine_annealing_initialization()
    test_cosine_annealing_epoch_zero()
    test_cosine_annealing_epoch_max()
    test_cosine_annealing_midpoint()
    test_cosine_annealing_smooth_decay()
    test_cosine_annealing_with_eta_min()
    test_cosine_annealing_different_t_max()
    test_cosine_annealing_beyond_t_max()

    print("\nAll scheduler tests (Part 1) passed! ✓")
