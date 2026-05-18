"""Tests for composite warmup schedulers.

Tests cover:
- WarmupCosineAnnealingLR: Combined warmup and cosine annealing
- WarmupStepLR: Combined warmup and step decay

All tests use the real scheduler implementations.
"""

from tests.projectodyssey.conftest import (
    assert_true,
    assert_equal,
    assert_almost_equal,
    assert_greater,
    assert_less_or_equal,
    TestFixtures,
)
from projectodyssey.training.schedulers import WarmupCosineAnnealingLR, WarmupStepLR


# ============================================================================
# WarmupCosineAnnealingLR Tests
# ============================================================================


def test_warmup_cosine_annealing_initialization() raises:
    """Test WarmupCosineAnnealingLR scheduler initialization."""
    var scheduler = WarmupCosineAnnealingLR(
        base_lr=0.1, warmup_epochs=10, T_max=100, eta_min=0.0
    )

    assert_almost_equal(scheduler.base_lr, 0.1)
    assert_equal(scheduler.warmup_epochs, 10)
    assert_equal(scheduler.T_max, 100)
    assert_almost_equal(scheduler.eta_min, 0.0)


def test_warmup_cosine_annealing_warmup_phase() raises:
    """Test WarmupCosineAnnealingLR during warmup phase.

    During warmup (0 to warmup_epochs), LR should increase linearly.
    """
    var scheduler = WarmupCosineAnnealingLR(
        base_lr=0.1, warmup_epochs=10, T_max=100, eta_min=0.0
    )

    # At epoch 0, LR should be 0
    var lr_0 = scheduler.get_lr(epoch=0)
    assert_almost_equal(lr_0, 0.0, tolerance=1e-6)

    # At epoch 5, LR should be halfway
    var lr_5 = scheduler.get_lr(epoch=5)
    assert_almost_equal(lr_5, 0.05, tolerance=1e-6)

    # At epoch 9 (just before warmup ends), LR should be 0.09
    var lr_9 = scheduler.get_lr(epoch=9)
    assert_almost_equal(lr_9, 0.09, tolerance=1e-6)


def test_warmup_cosine_annealing_after_warmup() raises:
    """Test WarmupCosineAnnealingLR after warmup phase.

    After warmup, LR should follow cosine annealing curve.
    """
    var scheduler = WarmupCosineAnnealingLR(
        base_lr=0.1, warmup_epochs=10, T_max=100, eta_min=0.0
    )

    # At epoch 10 (warmup ends), LR should equal base_lr
    var lr_10 = scheduler.get_lr(epoch=10)
    assert_almost_equal(lr_10, 0.1, tolerance=1e-6)

    # LR should decrease after warmup
    var lr_50 = scheduler.get_lr(epoch=50)
    assert_less_or_equal(lr_50, 0.1)

    # At final epoch, LR should approach eta_min
    var lr_100 = scheduler.get_lr(epoch=100)
    assert_almost_equal(lr_100, 0.0, tolerance=1e-6)


def test_warmup_cosine_annealing_monotonic_after_warmup() raises:
    """Test WarmupCosineAnnealingLR decreases monotonically after warmup."""
    var scheduler = WarmupCosineAnnealingLR(
        base_lr=0.1, warmup_epochs=10, T_max=100, eta_min=0.0
    )

    var previous_lr = scheduler.get_lr(10)
    for epoch in range(11, 101):
        var current_lr = scheduler.get_lr(epoch)
        assert_less_or_equal(current_lr, previous_lr)
        previous_lr = current_lr


# ============================================================================
# WarmupStepLR Tests
# ============================================================================


def test_warmup_step_lr_initialization() raises:
    """Test WarmupStepLR scheduler initialization."""
    var scheduler = WarmupStepLR(
        base_lr=0.1, warmup_epochs=10, step_size=30, gamma=0.1
    )

    assert_almost_equal(scheduler.base_lr, 0.1)
    assert_equal(scheduler.warmup_epochs, 10)
    assert_equal(scheduler.step_size, 30)
    assert_almost_equal(scheduler.gamma, 0.1)


def test_warmup_step_lr_warmup_phase() raises:
    """Test WarmupStepLR during warmup phase.

    During warmup (0 to warmup_epochs), LR should increase linearly.
    """
    var scheduler = WarmupStepLR(
        base_lr=0.1, warmup_epochs=10, step_size=30, gamma=0.1
    )

    # At epoch 0, LR should be 0
    var lr_0 = scheduler.get_lr(epoch=0)
    assert_almost_equal(lr_0, 0.0, tolerance=1e-6)

    # At epoch 5, LR should be halfway
    var lr_5 = scheduler.get_lr(epoch=5)
    assert_almost_equal(lr_5, 0.05, tolerance=1e-6)

    # At epoch 9, LR should be close to base_lr
    var lr_9 = scheduler.get_lr(epoch=9)
    assert_almost_equal(lr_9, 0.09, tolerance=1e-6)


def test_warmup_step_lr_step_decay_phase() raises:
    """Test WarmupStepLR during step decay phase.

    After warmup, LR should follow step decay schedule.
    """
    var scheduler = WarmupStepLR(
        base_lr=0.1, warmup_epochs=10, step_size=30, gamma=0.1
    )

    # At epoch 10 (warmup ends), LR should equal base_lr
    var lr_10 = scheduler.get_lr(epoch=10)
    assert_almost_equal(lr_10, 0.1, tolerance=1e-6)

    # Epochs 10-39 should stay at 0.1
    var lr_20 = scheduler.get_lr(epoch=20)
    assert_almost_equal(lr_20, 0.1, tolerance=1e-6)

    # At epoch 40 (first step after warmup), LR should be 0.01
    var lr_40 = scheduler.get_lr(epoch=40)
    assert_almost_equal(lr_40, 0.01, tolerance=1e-6)

    # At epoch 70 (second step), LR should be 0.001
    var lr_70 = scheduler.get_lr(epoch=70)
    assert_almost_equal(lr_70, 0.001, tolerance=1e-6)


def test_warmup_step_lr_step_interval() raises:
    """Test WarmupStepLR step decay intervals."""
    var scheduler = WarmupStepLR(
        base_lr=0.1, warmup_epochs=10, step_size=30, gamma=0.1
    )

    # Epochs 10-39: LR = 0.1
    for epoch in range(10, 40):
        var lr = scheduler.get_lr(epoch=epoch)
        assert_almost_equal(lr, 0.1, tolerance=1e-6)

    # Epochs 40-69: LR = 0.01
    for epoch in range(40, 70):
        var lr = scheduler.get_lr(epoch=epoch)
        assert_almost_equal(lr, 0.01, tolerance=1e-6)

    # Epochs 70+: LR = 0.001
    for epoch in range(70, 100):
        var lr = scheduler.get_lr(epoch=epoch)
        assert_almost_equal(lr, 0.001, tolerance=1e-6)


def main() raises:
    """Run all composite warmup scheduler tests."""
    print("Running WarmupCosineAnnealingLR tests...")
    test_warmup_cosine_annealing_initialization()
    test_warmup_cosine_annealing_warmup_phase()
    test_warmup_cosine_annealing_after_warmup()
    test_warmup_cosine_annealing_monotonic_after_warmup()

    print("Running WarmupStepLR tests...")
    test_warmup_step_lr_initialization()
    test_warmup_step_lr_warmup_phase()
    test_warmup_step_lr_step_decay_phase()
    test_warmup_step_lr_step_interval()

    print("All composite warmup scheduler tests passed! ✓")
