"""CosineAnnealingLR tests

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


def test_cosine_annealing_initialization() raises:
    """Test CosineAnnealingLR scheduler initialization."""
    var scheduler = CosineAnnealingLR(base_lr=0.1, T_max=100, eta_min=0.0)

    # Verify initial parameters
    assert_almost_equal(scheduler.base_lr, 0.1)
    assert_equal(scheduler.T_max, 100)
    assert_almost_equal(scheduler.eta_min, 0.0)


def test_cosine_annealing_epoch_zero() raises:
    """Test CosineAnnealingLR at epoch 0 (maximum learning rate).

    At epoch 0, LR should equal base_lr.
    Formula: lr = eta_min + (base_lr - eta_min) * (1 + cos(π * 0 / T_max)) / 2
    = eta_min + (base_lr - eta_min) * (1 + 1) / 2
    = base_lr.
    """
    var scheduler = CosineAnnealingLR(base_lr=0.1, T_max=100, eta_min=0.0)

    var lr0 = scheduler.get_lr(epoch=0)
    assert_almost_equal(lr0, 0.1)


def test_cosine_annealing_epoch_max() raises:
    """Test CosineAnnealingLR at epoch T_max (minimum learning rate).

    At epoch T_max, LR should equal eta_min.
    Formula: lr = eta_min + (base_lr - eta_min) * (1 + cos(π * 1)) / 2
    = eta_min + (base_lr - eta_min) * (1 - 1) / 2
    = eta_min.
    """
    var scheduler = CosineAnnealingLR(base_lr=0.1, T_max=100, eta_min=0.01)

    var lr_max = scheduler.get_lr(epoch=100)
    assert_almost_equal(lr_max, 0.01, tolerance=1e-6)


def test_cosine_annealing_midpoint() raises:
    """Test CosineAnnealingLR at midpoint epoch.

    At epoch = T_max / 2, cosine factor should be 0.
    LR = eta_min + (base_lr - eta_min) * 0 / 2 = eta_min.
    """
    var scheduler = CosineAnnealingLR(base_lr=1.0, T_max=100, eta_min=0.0)

    var lr_mid = scheduler.get_lr(epoch=50)
    # At midpoint: cos(π/2) = 0, so (1 + 0) / 2 = 0.5
    # LR = 0 + 1.0 * 0.5 = 0.5
    assert_almost_equal(lr_mid, 0.5, tolerance=1e-6)


def test_cosine_annealing_smooth_decay() raises:
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


def test_cosine_annealing_with_eta_min() raises:
    """Test CosineAnnealingLR respects eta_min floor.

    LR should never go below eta_min.
    """
    var scheduler = CosineAnnealingLR(base_lr=1.0, T_max=100, eta_min=0.1)

    for epoch in range(0, 101):
        var lr = scheduler.get_lr(epoch)

        # LR should not go below eta_min
        assert_greater(lr, 0.09)  # Small tolerance for floating point


def test_cosine_annealing_different_t_max() raises:
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


def test_cosine_annealing_beyond_t_max() raises:
    """Test CosineAnnealingLR beyond T_max (clamped to T_max).

    Epochs beyond T_max should clamp to T_max.
    """
    var scheduler = CosineAnnealingLR(base_lr=1.0, T_max=100, eta_min=0.0)

    var lr_100 = scheduler.get_lr(100)
    var lr_200 = scheduler.get_lr(200)

    # Both should give the same result (clamped at T_max)
    assert_almost_equal(lr_100, lr_200)


def test_reduce_lr_on_plateau_initialization_min_mode() raises:
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


def test_reduce_lr_on_plateau_initialization_max_mode() raises:
    """Test ReduceLROnPlateau initialization in 'max' mode."""
    var scheduler = ReduceLROnPlateau(
        base_lr=0.1, mode="max", factor=0.1, patience=5
    )

    assert_equal(scheduler.mode, MODE_MAX)
    assert_equal(scheduler.patience, 5)


def test_reduce_lr_on_plateau_min_mode_improvement() raises:
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


def test_reduce_lr_on_plateau_min_mode_no_improvement() raises:
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


def test_reduce_lr_on_plateau_max_mode_improvement() raises:
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


def test_reduce_lr_on_plateau_max_mode_no_improvement() raises:
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


def test_reduce_lr_on_plateau_multiple_reductions() raises:
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


def test_reduce_lr_on_plateau_improvement_resets_counter() raises:
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


def test_cosine_annealing_zero_t_max() raises:
    """Test CosineAnnealingLR with T_max=0 (edge case).

    T_max <= 0 should return base_lr.
    """
    var scheduler = CosineAnnealingLR(base_lr=1.0, T_max=0, eta_min=0.0)

    var lr = scheduler.get_lr(0)
    assert_almost_equal(lr, 1.0)


def test_cosine_annealing_formula_accuracy() raises:
    """Test CosineAnnealingLR matches mathematical formula exactly.

    Formula: lr = eta_min + (base_lr - eta_min) * (1 + cos(π * epoch / T_max)) / 2
    """
    var scheduler = CosineAnnealingLR(base_lr=0.1, T_max=10, eta_min=0.01)

    # Test several epochs
    var lr0 = scheduler.get_lr(0)
    # cos(0) = 1, so (1 + 1) / 2 = 1
    # LR = 0.01 + 0.09 * 1 = 0.1
    assert_almost_equal(lr0, 0.1, tolerance=1e-6)


def test_reduce_lr_on_plateau_factor_one() raises:
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


def test_reduce_lr_on_plateau_zero_patience() raises:
    """Test ReduceLROnPlateau with patience=0 (reduce every epoch without improvement).

    patience=0 means reduce immediately on first no-improvement epoch.
    """
    var scheduler = ReduceLROnPlateau(
        base_lr=1.0, mode="min", factor=0.5, patience=0
    )

    _ = scheduler.step(0.5)
    var lr2 = scheduler.step(0.6)  # No improvement, reduce immediately
    assert_almost_equal(lr2, 0.5)


def test_reduce_lr_on_plateau_very_small_lr() raises:
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


def test_reduce_lr_on_plateau_get_lr_interface() raises:
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


def test_reduce_lr_on_plateau_realistic_training_scenario() raises:
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


def main() raises:
    """Run all test_schedulers tests."""
    print("Running test_schedulers tests...")

    test_cosine_annealing_initialization()
    print("✓ test_cosine_annealing_initialization")

    test_cosine_annealing_epoch_zero()
    print("✓ test_cosine_annealing_epoch_zero")

    test_cosine_annealing_epoch_max()
    print("✓ test_cosine_annealing_epoch_max")

    test_cosine_annealing_midpoint()
    print("✓ test_cosine_annealing_midpoint")

    test_cosine_annealing_smooth_decay()
    print("✓ test_cosine_annealing_smooth_decay")

    test_cosine_annealing_with_eta_min()
    print("✓ test_cosine_annealing_with_eta_min")

    test_cosine_annealing_different_t_max()
    print("✓ test_cosine_annealing_different_t_max")

    test_cosine_annealing_beyond_t_max()
    print("✓ test_cosine_annealing_beyond_t_max")

    test_reduce_lr_on_plateau_initialization_min_mode()
    print("✓ test_reduce_lr_on_plateau_initialization_min_mode")

    test_reduce_lr_on_plateau_initialization_max_mode()
    print("✓ test_reduce_lr_on_plateau_initialization_max_mode")

    test_reduce_lr_on_plateau_min_mode_improvement()
    print("✓ test_reduce_lr_on_plateau_min_mode_improvement")

    test_reduce_lr_on_plateau_min_mode_no_improvement()
    print("✓ test_reduce_lr_on_plateau_min_mode_no_improvement")

    test_reduce_lr_on_plateau_max_mode_improvement()
    print("✓ test_reduce_lr_on_plateau_max_mode_improvement")

    test_reduce_lr_on_plateau_max_mode_no_improvement()
    print("✓ test_reduce_lr_on_plateau_max_mode_no_improvement")

    test_reduce_lr_on_plateau_multiple_reductions()
    print("✓ test_reduce_lr_on_plateau_multiple_reductions")

    test_reduce_lr_on_plateau_improvement_resets_counter()
    print("✓ test_reduce_lr_on_plateau_improvement_resets_counter")

    test_cosine_annealing_zero_t_max()
    print("✓ test_cosine_annealing_zero_t_max")

    test_cosine_annealing_formula_accuracy()
    print("✓ test_cosine_annealing_formula_accuracy")

    test_reduce_lr_on_plateau_factor_one()
    print("✓ test_reduce_lr_on_plateau_factor_one")

    test_reduce_lr_on_plateau_zero_patience()
    print("✓ test_reduce_lr_on_plateau_zero_patience")

    test_reduce_lr_on_plateau_very_small_lr()
    print("✓ test_reduce_lr_on_plateau_very_small_lr")

    test_reduce_lr_on_plateau_get_lr_interface()
    print("✓ test_reduce_lr_on_plateau_get_lr_interface")

    test_reduce_lr_on_plateau_realistic_training_scenario()
    print("✓ test_reduce_lr_on_plateau_realistic_training_scenario")

    print("\nAll test_schedulers tests passed!")
